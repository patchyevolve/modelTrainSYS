import sys
import torch
import torch.nn.functional as F

_CPP_PATH = __file__.rsplit("/", 2)[0] + "/csrc"
if _CPP_PATH not in sys.path:
    sys.path.insert(0, _CPP_PATH)

try:
    import logit_processors as _cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False


def apply_repetition_penalty(logits: torch.Tensor, ids, penalty: float, window: int):
    if CPP_AVAILABLE:
        _cpp.apply_repetition_penalty(logits.cpu().numpy().astype('float32'),
                                       ids, penalty, window)
        return
    if penalty == 1.0 or not ids:
        return
    for pid in set(ids[-window:]):
        if 0 <= pid < logits.size(-1):
            logits[pid] /= penalty


def apply_top_k(logits: torch.Tensor, k: int):
    if CPP_AVAILABLE:
        _cpp.apply_top_k(logits.cpu().numpy().astype('float32'), k)
        return
    if k <= 0:
        return
    kth = torch.topk(logits, min(k, logits.size(-1))).values[-1]
    logits.masked_fill_(logits < kth, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: float):
    if CPP_AVAILABLE:
        _cpp.apply_top_p(logits.cpu().numpy().astype('float32'), p)
        return
    if not (0.0 < p < 1.0):
        return
    sl, si = torch.sort(logits, descending=True)
    cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
    sl[cp - F.softmax(sl, dim=-1) > p] = float("-inf")
    logits.copy_(torch.zeros_like(logits).scatter_(0, si, sl))


def apply_min_p(logits: torch.Tensor, min_p: float):
    if CPP_AVAILABLE:
        _cpp.apply_min_p(logits.cpu().numpy().astype('float32'), min_p)
        return
    if min_p <= 0.0:
        return
    top_prob = F.softmax(logits, dim=-1).max()
    threshold = top_prob * min_p
    logits[F.softmax(logits, dim=-1) < threshold] = float("-inf")


def sample(logits: torch.Tensor) -> int:
    if CPP_AVAILABLE:
        return _cpp.sample(logits.cpu().numpy().astype('float32'))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()
