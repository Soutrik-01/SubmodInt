import torch
import math
from tqdm import tqdm
import os 

@torch.jit.script
def get_square(mat):
    return mat**2


def get_cdist(V):
    dist_mat = torch.cdist(V, V)
    return get_square(dist_mat)


def get_rbf_kernel(dist_mat, kw):
    sim = torch.exp(-dist_mat/(kw*dist_mat.mean()))
    return sim


def FL_evaluation(sim, sets, norm=1.):
    foo = sim.unsqueeze(0)[:, sets].squeeze()
    assert foo.shape[0] == len(sets)
    foo = ((torch.amax(foo, dim=1)).squeeze()).sum(dim=-1)/norm
    return foo

def get_features(model, X, device, bsz=2048):
    h = []
    num_batches = math.ceil(len(X)/bsz)
    with torch.no_grad():
        model.eval()
        for i in (range(num_batches)):
            input = X[i*bsz: (i+1)*bsz].to(device).float()
            _, output = model(input, last=True)

            h.append(output)

    h = torch.cat(h, dim=0)
    return h


class facilityLocation:
    def __init__(self, V, device, args, labels=None):
        self.sim = None
        if args.get('fl_on_the_fly'):
            self.device = device
            self.V = V.squeeze().float().to(device)
            self.sq = (self.V ** 2).sum(-1)
            self.scale = args['kw'] * 2 * (self.sq.mean() - (self.V.mean(0) ** 2).sum())
            return
        if args['dset'] == 'IN100':    
            import numpy as np
            path = os.path.join(os.path.join(args["root"], args["dset"]), "similarity")
            N = 130000
            num_shards = 65
            self.sim = []
            print("Loading matrix in shards..")
            for ii in tqdm(range(num_shards)):
                self.sim.append(np.load(f"{path}/IN100_sim_train_sharded_{ii}.npy"))
            print("Loaded all shards. Concatenating now..")
            self.sim = np.concatenate(self.sim, axis=0)
            self.sim =  torch.from_numpy(self.sim)

        self.device = device
        kw = args['kw']
        if self.sim is None:
            V = V.to(self.device)
            print(f"Generating {len(V)},{len(V)} RBF similarity matrix")
            self.sim = get_rbf_kernel(get_cdist(V.squeeze()), kw).cpu()
            V = None
            del (V)
            torch.cuda.empty_cache()

    def get_FL_evals_nested(self, D_M_full, nesting_list, MAX=0, SCMI=False, bsz=2048):
        print(self.device)
        self.sim = self.sim.to(self.device)
        f_V = len(self.sim)

        N = len(D_M_full)
        num_batches = N//bsz if N % bsz == 0 else N//bsz + 1

        eval_full = [[] for i in range(len(nesting_list))]

        for i in (range(num_batches)):
            A = D_M_full[i*(bsz):(i+1)*bsz]

            for j in range(len(nesting_list)):
                A_ = A[:, :nesting_list[j]]
                n = len(A_)
                FL_M = FL_evaluation(self.sim, A_, f_V)

                eval_full[j].append(FL_M)

        eval_full = [torch.cat(m, dim=-1) for m in eval_full]
        eval_full = torch.stack(eval_full, dim=0)
        

        self.sim = self.sim.cpu()
        eval_full = eval_full.cpu()
        torch.cuda.empty_cache()
        return eval_full

    def kernel(self, rows):
        """RBF similarities between items `rows` and all n items, shape (len(rows), n)."""
        d2 = (self.sq[rows].unsqueeze(1) + self.sq.unsqueeze(0) - 2 * self.V[rows] @ self.V.T).clamp(min=0)
        return torch.exp(-d2 / self.scale)

    def nested_fl(self, sets, nesting_list, chunk=16):
        """FL(sets[:, :k]) / n for each k in nesting_list, shape (len(nesting_list), len(sets))."""
        cols = torch.tensor(nesting_list, device=self.device) - 1
        out = []
        for part in sets.to(self.device).long().split(chunk):
            sim = self.kernel(part.reshape(-1)).view(*part.shape, -1)
            out.append(sim.cummax(dim=1).values[:, cols].sum(-1))
        return torch.cat(out).T / len(self.V)

    def get_margin_FL_nested_on_the_fly(self, D_M_full, D_E_full, nesting_list, Y, matroid_rank, rank_tradeoff, target_responsibility, bsz):
        margins, fl_m, fl_e = [], [], []
        for i in tqdm(range(0, len(D_M_full), bsz)):
            A, B = D_M_full[i:i + bsz], D_E_full[i:i + bsz]
            n = len(A)
            sets = torch.cat([A, B])[:, :max(nesting_list)]
            values = self.nested_fl(sets, nesting_list)
            if matroid_rank > -1:
                labels = Y.to(self.device)[sets.to(self.device).long()].sort(dim=1).values
                new_class = torch.cat([torch.ones_like(labels[:, :1]), (labels.diff(dim=1) != 0).long()], dim=1)
                rank = torch.stack([new_class[:, :k].sum(1) for k in nesting_list]).float() / 100
            else:
                rank = torch.zeros_like(values)
            margins.append(target_responsibility * (values[:, n:] - values[:, :n]) + rank_tradeoff * (rank[:, n:] - rank[:, :n]))
            fl_m.append(target_responsibility * values[:, :n] + rank_tradeoff * rank[:, :n])
            fl_e.append(target_responsibility * values[:, n:] + rank_tradeoff * rank[:, n:])
        return torch.cat(margins, 1).cpu(), torch.cat(fl_m, 1).cpu(), torch.cat(fl_e, 1).cpu()

    def get_margin_FL_nested(self, D_M_full, D_E_full, nesting_list, Y=None, matroid_rank=-1, rank_tradeoff=0., target_responsibility=1., bsz=40):
        if self.sim is None:
            return self.get_margin_FL_nested_on_the_fly(D_M_full, D_E_full, nesting_list, Y, matroid_rank, rank_tradeoff, target_responsibility, bsz)
        self.sim = self.sim.to(self.device)
        f_V = torch.max(self.sim, dim=-1)[0].sum()
        if matroid_rank > -1:
            Y = Y.to(self.device)

        N = len(D_M_full)
        num_batches = N//bsz if N % bsz == 0 else N//bsz + 1

        margin_full = [[] for i in range(len(nesting_list))]
        FL_M_full = [[] for i in range(len(nesting_list))]
        FL_E_full = [[] for i in range(len(nesting_list))]

        for i in tqdm(range(num_batches)):
            A = D_M_full[i*(bsz):(i+1)*bsz]
            B = D_E_full[i*(bsz):(i+1)*bsz]

            for j in range(len(nesting_list)):
                A_ = A[:, :nesting_list[j]]
                B_ = B[:, :nesting_list[j]]
                n = len(A_)
                FL_M_E = FL_evaluation(
                    self.sim, torch.cat([A_, B_], dim=0), f_V)


                if matroid_rank > -1: 
                    Y_M_E = Y[torch.cat([A_, B_])]
                    rank_M_E = (torch.Tensor([len(torch.unique(Y_M_E[i])) for i in range(
                        len(FL_M_E))])/100).to(self.device, non_blocking=True)
                else:
                    rank_M_E = torch.zeros(len(FL_M_E), device=self.device)
                margin_full[j].append(
                    target_responsibility*(FL_M_E[n:]-FL_M_E[:n]) + rank_tradeoff*(rank_M_E[n:]-rank_M_E[:n]))
                FL_M_full[j].append(
                    target_responsibility*FL_M_E[:n] + (rank_tradeoff*rank_M_E)[:n])
                FL_E_full[j].append(
                    target_responsibility*FL_M_E[n:] + (rank_tradeoff*rank_M_E)[n:])

        margin_full = [torch.cat(m, dim=-1) for m in margin_full]
        FL_M_full = [torch.cat(m, dim=-1) for m in FL_M_full]
        FL_E_full = [torch.cat(m, dim=-1) for m in FL_E_full]

        margin_full = torch.stack(margin_full, dim=0)
        FL_M_full = torch.stack(FL_M_full, dim=0)
        FL_E_full = torch.stack(FL_E_full, dim=0)

        self.sim = self.sim.cpu()
        margin_full = margin_full.cpu()
        FL_M_full = FL_M_full.cpu()
        FL_E_full = FL_E_full.cpu()
        torch.cuda.empty_cache()
        return margin_full, FL_M_full, FL_E_full
