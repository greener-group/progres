# Fast protein structure searching using structure graph embeddings

# faiss and biopython imported in functions
import torch
from torch.nn import Dropout, Identity, Linear, Sequential, SiLU
from torch.nn.functional import normalize
from torch.utils.data import Dataset
from torch_scatter import scatter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from einops import rearrange
import importlib.metadata
from math import ceil
import os
import sys
from urllib import request

n_layers = 6
embedding_size = 128
hidden_dim = 128
hidden_egnn_dim = 64
hidden_edge_dim = 256
pos_embed_dim = 64
n_features = pos_embed_dim + 4
pos_embed_freq_inv = 2000
contact_dist = 10.0 # Å
dropout = 0.0
dropout_final = 0.0
default_minsimilarity = 0.8
default_maxhits = 100
pre_embedded_dbs = ["scope95", "scope40", "cath40", "ecod70", "af21org"]
pre_embedded_dbs_faiss = ["afted"]
zenodo_record = "10975201" # This only needs to change when the trained model or databases change
trained_model_subdir = "v_0_2_0" # This only needs to change when the trained model changes
database_subdir      = "v_0_2_1" # This only needs to change when the databases change
progres_dir       = os.path.dirname(os.path.realpath(__file__))
trained_model_dir = os.path.join(progres_dir, "trained_models", trained_model_subdir)
database_dir      = os.path.join(progres_dir, "databases"     , database_subdir     )
trained_model_fp  = os.path.join(trained_model_dir, "trained_model.pt")

class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        channels = int(ceil(channels / 2) * 2)
        inv_freq = 1.0 / (pos_embed_freq_inv ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        sin_inp_x = torch.einsum("...i,j->...ij", tensor, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x

pos_embedder = SinusoidalPositionalEncoding(pos_embed_dim)

def batched_index_select(values, indices, dim=1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

# From https://github.com/lucidrains/egnn-pytorch
class EGNN(torch.nn.Module):
    def __init__(
        self,
        dim,
        m_dim=16,
        dropout=0.0,
        init_eps=1e-3,
    ):
        super().__init__()
        edge_input_dim = (dim * 2) + 1
        dropout = Dropout(dropout) if dropout > 0 else Identity()

        self.edge_mlp = Sequential(
            Linear(edge_input_dim, hidden_edge_dim),
            dropout,
            SiLU(),
            Linear(hidden_edge_dim, m_dim),
            SiLU(),
        )

        self.node_mlp = Sequential(
            Linear(dim + m_dim, dim * 2),
            dropout,
            SiLU(),
            Linear(dim * 2, dim),
        )

        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {Linear}:
            torch.nn.init.normal_(module.weight, std=self.init_eps)

    def forward(self, feats, coors, mask, adj_mat):
        b, n, d, device = *feats.shape, feats.device

        rel_coors = rearrange(coors, "b i d -> b i () d") - rearrange(coors, "b j d -> b () j d")
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        i = j = n
        ranking = rel_dist[..., 0].clone()
        rank_mask = mask[:, :, None] * mask[:, None, :]
        ranking.masked_fill_(~rank_mask, 1e5)

        num_nearest = int(adj_mat.float().sum(dim=-1).max().item())
        valid_radius = 0

        self_mask = rearrange(torch.eye(n, device=device, dtype=torch.bool), "i j -> () i j")

        adj_mat = adj_mat.masked_fill(self_mask, False)
        ranking.masked_fill_(self_mask, -1.)
        ranking.masked_fill_(adj_mat, 0.)

        nbhd_ranking, nbhd_indices = ranking.topk(num_nearest, dim=-1, largest=False)
        nbhd_mask = nbhd_ranking <= valid_radius

        rel_coors = batched_index_select(rel_coors, nbhd_indices, dim=2)
        rel_dist = batched_index_select(rel_dist, nbhd_indices, dim=2)

        j = num_nearest
        feats_j = batched_index_select(feats, nbhd_indices, dim=1)
        feats_i = rearrange(feats, "b i d -> b i () d")
        feats_i, feats_j = torch.broadcast_tensors(feats_i, feats_j)

        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim=-1)
        m_ij = self.edge_mlp(edge_input)

        mask_i = rearrange(mask, "b i -> b i ()")
        mask_j = batched_index_select(mask, nbhd_indices, dim = 1)
        mask = (mask_i * mask_j) & nbhd_mask

        m_ij_mask = rearrange(mask, "... -> ... ()")
        m_ij = m_ij.masked_fill(~m_ij_mask, 0.)
        m_i = m_ij.sum(dim=-2)

        node_mlp_input = torch.cat((feats, m_i), dim = -1)
        node_out = self.node_mlp(node_mlp_input) + feats

        return node_out, coors

# Based on https://github.com/vgsatorras/egnn/blob/main/qm9/models.py
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_enc = Linear(n_features, hidden_dim)
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(EGNN(
                dim=hidden_dim,
                m_dim=hidden_egnn_dim,
                dropout=dropout,
            ))
        self.node_dec = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            Dropout(dropout) if dropout > 0 else Identity(),
            SiLU(),
            Linear(hidden_dim, hidden_dim),
        )
        self.graph_dec = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            Dropout(dropout) if dropout > 0 else Identity(),
            SiLU(),
            Dropout(dropout_final) if dropout_final > 0 else Identity(),
            Linear(hidden_dim, embedding_size),
        )

    def forward(self, data):
        device = data.x.device
        feats, coords = data.x.unsqueeze(0), data.coords.unsqueeze(0)
        adj_mat = torch.sparse_coo_tensor(
            indices=data.edge_index,
            values=torch.tensor([1] * data.edge_index.size(1), device=device),
            size=(data.num_nodes, data.num_nodes),
        ).to_dense().bool().unsqueeze(0)
        mask = torch.ones(1, data.num_nodes, dtype=torch.bool, device=device)
        feats = self.node_enc(feats)
        for layer in self.layers:
            feats, coords = layer(feats, coords, mask, adj_mat)
        feats = self.node_dec(feats)
        batch = torch.tensor([0] * data.num_nodes, device=device) if data.batch is None else data.batch
        graph_feats = scatter(feats.squeeze(0), batch, dim=0, reduce="sum")
        out = self.graph_dec(graph_feats)
        return normalize(out, dim=1)

# Running the model in  __getitem__ allows multiple workers to be used on CPU
class StructureDataset(Dataset):
    def __init__(self, file_paths, fileformat, model, device):
        self.file_paths = file_paths
        self.fileformat = fileformat
        self.model = model
        self.device = device

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        graph = read_graph(self.file_paths[idx], self.fileformat)
        emb = self.model(graph.to(self.device)).squeeze(0)
        nres = torch.tensor([graph.num_nodes], device=self.device)
        return emb, nres

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings):
        err_msg = f"embeddings must have shape ({embedding_size}) or (n, {embedding_size})"
        if embeddings.dim() == 1:
            if embeddings.size(0) != embedding_size:
                raise ValueError(err_msg)
            self.embeddings = embeddings.unsqueeze(0)
        else:
            if embeddings.size(1) != embedding_size:
                raise ValueError(err_msg)
            self.embeddings = embeddings

    def __len__(self):
        return self.embeddings.size(0)

    def __getitem__(self, idx):
        emb = self.embeddings[idx]
        nres = "?"
        return emb, nres

def read_coords(fp, fileformat="guess"):
    if fileformat == "guess":
        chosen_format = "pdb"
        file_ext = os.path.splitext(fp)[1].lower()
        if file_ext == ".cif" or file_ext == ".mmcif":
            chosen_format = "mmcif"
        elif file_ext == ".mmtf":
            chosen_format = "mmtf"
    else:
        chosen_format = fileformat

    coords = []
    if chosen_format == "pdb":
        with open(fp) as f:
            for line in f.readlines():
                if (line.startswith("ATOM  ") or line.startswith("HETATM")) and line[12:16].strip() == "CA":
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                elif line.startswith("ENDMDL"):
                    break
    elif chosen_format == "mmcif" or chosen_format == "mmtf":
        if chosen_format == "mmcif":
            from Bio.PDB.MMCIFParser import MMCIFParser
            parser = MMCIFParser()
            struc = parser.get_structure("", fp)
        else:
            from Bio.PDB.mmtf import MMTFParser
            struc = MMTFParser.get_structure(fp)
        for model in struc:
            for atom in model.get_atoms():
                if atom.get_name() == "CA":
                    cs = atom.get_coord()
                    coords.append([float(cs[0]), float(cs[1]), float(cs[2])])
            break
    elif chosen_format == "coords":
        with open(fp) as f:
            for line in f.readlines():
                coords.append([float(v) for v in line.rstrip().split()])
    else:
        raise ValueError("fileformat must be \"guess\", \"pdb\", \"mmcif\", \"mmtf\" or \"coords\"")
    return coords

def coords_to_graph(coords):
    n_res = len(coords)
    if not isinstance(coords, torch.Tensor):
        coords = torch.tensor(coords)
    if coords.size(1) != 3:
        raise ValueError("coords must be, or must be convertible to, a tensor of shape (nres, 3)")
    dmap = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0),
                       compute_mode="donot_use_mm_for_euclid_dist")
    contacts = (dmap <= contact_dist).squeeze(0)
    edge_index = contacts.to_sparse().indices()

    degrees = contacts.sum(dim=0)
    norm_degrees = (degrees / degrees.max()).unsqueeze(1)
    term_features = [[0.0, 0.0] for _ in range(n_res)]
    term_features[ 0][0] = 1.0
    term_features[-1][1] = 1.0
    term_features = torch.tensor(term_features)

    # The tau torsion angle is between 4 consecutive Cα atoms, we assign it to the second Cα
    # This feature breaks mirror invariance
    vec_ab = coords[1:-2] - coords[ :-3]
    vec_bc = coords[2:-1] - coords[1:-2]
    vec_cd = coords[3:  ] - coords[2:-1]
    cross_ab_bc = torch.cross(vec_ab, vec_bc, dim=1)
    cross_bc_cd = torch.cross(vec_bc, vec_cd, dim=1)
    taus = torch.atan2(
        (torch.cross(cross_ab_bc, cross_bc_cd, dim=1) * normalize(vec_bc, dim=1)).sum(dim=1),
        (cross_ab_bc * cross_bc_cd).sum(dim=1),
    )
    taus_pad = torch.cat((
        torch.tensor([0.0]),
        taus / torch.pi, # Convert to range -1 -> 1
        torch.tensor([0.0, 0.0]),
    )).unsqueeze(1)

    pos_embed = pos_embedder(torch.arange(1, n_res + 1))
    x = torch.cat((norm_degrees, term_features, taus_pad, pos_embed), dim=1)
    data = Data(x=x, edge_index=edge_index, coords=coords)
    return data

def read_graph(fp, fileformat="guess"):
    coords = read_coords(fp, fileformat)
    return coords_to_graph(coords)

def embedding_distance(emb_1, emb_2):
    cosine_dist = (emb_1 * emb_2).sum(dim=-1) # Normalised in the model
    return (1 - cosine_dist) / 2 # Runs 0 (close) to 1 (far)

def embedding_similarity(emb_1, emb_2):
    cosine_dist = (emb_1 * emb_2).sum(dim=-1) # Normalised in the model
    return (1 + cosine_dist) / 2 # Runs 0 (far) to 1 (close)

def load_trained_model(device="cpu"):
    model = Model().to(device)
    loaded_model = torch.load(trained_model_fp, map_location=device)
    model.load_state_dict(loaded_model["model"])
    model.eval()
    return model

def embed_graph(graph, device="cpu", model=None):
    with torch.no_grad():
        if model is None:
            model = load_trained_model(device)
        data_loader = DataLoader([graph], batch_size=1)
        for batch in data_loader:
            emb = model(batch.to(device))
            break
        return emb.squeeze(0)

def embed_coords(coords, device="cpu", model=None):
    graph = coords_to_graph(coords)
    return embed_graph(graph, device, model)

def embed_structure(querystructure, fileformat="guess", device="cpu", model=None):
    graph = read_graph(querystructure, fileformat)
    return embed_graph(graph, device, model)

def get_batch_size(device="cpu", using_faiss=False):
    if using_faiss:
        return 64
    else:
        return 8

def get_num_workers(device="cpu"):
    if device == "cpu":
        return torch.get_num_threads()
    else:
        return 0

def download_data_if_required(download_afted=False):
    url_base = f"https://zenodo.org/record/{zenodo_record}/files"
    fps = [trained_model_fp]
    urls = [f"{url_base}/trained_model.pt"]
    for targetdb in pre_embedded_dbs:
        fps.append(os.path.join(database_dir, targetdb + ".pt"))
        urls.append(f"{url_base}/{targetdb}.pt")
    if download_afted:
        for fn in ["afted.index", "afted_noembs.pt"]:
            fps.append(os.path.join(database_dir, fn))
            urls.append(f"{url_base}/{fn}")

    if not os.path.isdir(trained_model_dir):
        os.makedirs(trained_model_dir)
    if not os.path.isdir(database_dir):
        os.makedirs(database_dir)

    printed = False
    for fp, url in zip(fps, urls):
        if not os.path.isfile(fp):
            if fp.endswith("afted.index"):
                print("Downloading afted data as first time setup (~33 GB) to ", database_dir,
                      ", internet connection required, this may take a while",
                      sep="", file=sys.stderr)
                printed = True
            if not printed:
                print("Downloading data as first time setup (~220 MB) to ", progres_dir,
                      ", internet connection required, this can take a few minutes",
                      sep="", file=sys.stderr)
                printed = True
            try:
                request.urlretrieve(url, fp)
                if not fp.endswith("afted.index"):
                    d = torch.load(fp, map_location="cpu")
                    if fp == trained_model_fp:
                        assert "model" in d
                    else:
                        assert "notes" in d
            except:
                if os.path.isfile(fp):
                    os.remove(fp)
                print("Failed to download from", url, "and save to", fp, file=sys.stderr)
                print("Exiting", file=sys.stderr)
                sys.exit(1)

    if printed:
        print("Data downloaded successfully", file=sys.stderr)

def progres_search_generator(querystructure=None, querylist=None, queryembeddings=None,
                             targetdb=None, fileformat="guess", minsimilarity=default_minsimilarity,
                             maxhits=default_maxhits, device="cpu", batch_size=None):
    if querystructure is None and querylist is None and queryembeddings is None:
        raise ValueError("One of querystructure, querylist or queryembeddings must be given")
    if targetdb is None:
        raise ValueError("targetdb must be given")

    download_data_if_required(targetdb == "afted")

    if targetdb in pre_embedded_dbs_faiss:
        import faiss
        print(f"Loading {targetdb} data, this can take a minute", file=sys.stderr)
        target_index = faiss.read_index(os.path.join(database_dir, f"{targetdb}.index"))
        target_data = torch.load(os.path.join(database_dir, f"{targetdb}_noembs.pt"), map_location=device)
        search_type = "faiss"
    elif targetdb in pre_embedded_dbs:
        target_fp = os.path.join(database_dir, targetdb + ".pt")
        target_data = torch.load(target_fp, map_location=device)
        search_type = "torch"
    else:
        target_data = torch.load(targetdb, map_location=device)
        search_type = "torch"

    model = load_trained_model(device)
    if querystructure is not None:
        query_fps = [querystructure]
        data_set = StructureDataset(query_fps, fileformat, model, device)
        num_workers = get_num_workers(device)
    elif querylist is not None:
        query_fps = []
        with open(querylist) as f:
            for line in f.readlines():
                query_fps.append(line.rstrip())
        data_set = StructureDataset(query_fps, fileformat, model, device)
        num_workers = get_num_workers(device)
    else:
        data_set = EmbeddingDataset(queryembeddings.to(device))
        query_fps = ["?"] * len(data_set)
        num_workers = 0

    if batch_size is None:
        batch_size = get_batch_size(device, search_type == "faiss")
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    with torch.no_grad():
        qi = 0
        for embs, nress in data_loader:
            if search_type == "faiss":
                sims_ord_batch, inds_ord_batch = target_index.search(embs.cpu().numpy(), maxhits)

            for bi in range(embs.size(0)):
                query_size = nress[bi].item() if type(nress) == torch.Tensor else "?"
                if search_type == "faiss":
                    sims_ord = sims_ord_batch[bi]
                    inds_ord = inds_ord_batch[bi]
                else:
                    dists = embedding_distance(embs[bi], target_data["embeddings"])
                    inds_ord = dists.argsort()[:maxhits].to("cpu")
                domids, hits_nres, similarities, notes = [], [], [], []
                if search_type == "faiss":
                    for si, similarity in enumerate(sims_ord):
                        if similarity < minsimilarity:
                            break
                        domids.append(target_data["ids"][inds_ord[si]])
                        hits_nres.append(target_data["nres"][inds_ord[si]])
                        similarities.append(similarity)
                        notes.append(target_data["notes"][inds_ord[si]])
                else:
                    for i in inds_ord:
                        similarity = (1 - dists[i]).item()
                        if similarity < minsimilarity:
                            break
                        domids.append(target_data["ids"][i])
                        hits_nres.append(target_data["nres"][i])
                        similarities.append(similarity)
                        notes.append(target_data["notes"][i])

                result_dict = {
                    "query_num":     qi + 1,
                    "query":         query_fps[qi],
                    "query_size":    query_size,
                    "database":      targetdb,
                    "minsimilarity": minsimilarity,
                    "maxhits":       maxhits,
                    "domains":       domids,
                    "hits_nres":     hits_nres,
                    "similarities":  similarities,
                    "notes":         notes,
                }
                qi += 1

                yield result_dict

def progres_search(querystructure=None, querylist=None, queryembeddings=None, targetdb=None,
                   fileformat="guess", minsimilarity=default_minsimilarity, maxhits=default_maxhits,
                   device="cpu", batch_size=None):
    generator = progres_search_generator(querystructure, querylist, queryembeddings, targetdb,
                                         fileformat, minsimilarity, maxhits, device, batch_size)
    return list(generator)

def progres_search_print(querystructure=None, querylist=None, queryembeddings=None, targetdb=None,
                         fileformat="guess", minsimilarity=default_minsimilarity,
                         maxhits=default_maxhits, device="cpu", batch_size=None):
    generator = progres_search_generator(querystructure, querylist, queryembeddings, targetdb,
                                         fileformat, minsimilarity, maxhits, device, batch_size)
    version = importlib.metadata.version("progres")

    for rd in generator:
        n_hits = len(rd["domains"])
        inds_str = [str(i + 1) for i in range(n_hits)]
        hits_nres_str = [str(n) for n in rd["hits_nres"]]
        padding_inds      = max(len(s) for s in inds_str        + ["# HIT_N" ])
        padding_domids    = max(len(s) for s in rd["domains"]   + ["DOMAIN"  ])
        padding_hits_nres = max(len(s) for s in hits_nres_str   + ["HIT_NRES"])
        faiss_str = ", FAISS search" if targetdb in pre_embedded_dbs_faiss else ""

        print("# QUERY_NUM:" , rd["query_num"])
        print("# QUERY:"     , rd["query"])
        print("# QUERY_SIZE:", rd["query_size"], "residues")
        print("# DATABASE:", targetdb)
        print(f"# PARAMETERS: minsimilarity {minsimilarity}, maxhits {maxhits}, progres v{version}{faiss_str}")
        print("  ".join([
            "#" + " HIT_N".rjust(padding_inds - 1),
            "DOMAIN".ljust(padding_domids),
            "HIT_NRES".rjust(padding_hits_nres),
            "SIMILARITY",
            "NOTES",
        ]))
        for i in range(n_hits):
            print("  ".join([
                inds_str[i].rjust(padding_inds),
                rd["domains"][i].ljust(padding_domids),
                hits_nres_str[i].rjust(padding_hits_nres),
                f"{rd['similarities'][i]:10.4f}",
                rd["notes"][i],
            ]))
        print()

def progres_embed(structurelist, outputfile, fileformat="guess", device="cpu", batch_size=None):
    download_data_if_required()

    fps, domids, notes = [], [], []
    with open(structurelist) as f:
        for line in f.readlines():
            cols = line.strip().split(None, 2)
            fps.append(cols[0])
            domids.append(cols[1])
            notes.append(cols[2] if len(cols) > 2 else "-")

    model = load_trained_model(device)
    data_set = StructureDataset(fps, fileformat, model, device)
    if batch_size is None:
        batch_size = get_batch_size(device)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=get_num_workers(device),
    )

    with torch.no_grad():
        embeddings = torch.zeros(len(data_set), embedding_size, device=device)
        n_residues = torch.zeros(len(data_set), dtype=torch.int, device=device)
        for bi, (embs, nress) in enumerate(data_loader):
            embeddings[(bi * batch_size):(bi * batch_size + embs.size(0))] = embs
            n_residues[(bi * batch_size):(bi * batch_size + embs.size(0))] = nress.squeeze(1)

    torch.save(
        {
            "ids"       : domids,
            "embeddings": embeddings.cpu().to(torch.float16),
            "nres"      : list(n_residues.cpu().numpy()),
            "notes"     : notes,
        },
        outputfile,
    )
