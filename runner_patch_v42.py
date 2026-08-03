"""V4.2 patch: balance source/class groups across small micro-batches.

The V4.1 T4 profile uses batch_size=2 and grad_accum=2.  A training fold has
four source/class groups (two sources x two classes), so requiring every
micro-batch to contain all four groups is impossible.  This patch rotates the
groups across consecutive micro-batches while greedily maximizing source and
class diversity inside each micro-batch.
"""


def _replace_once(code: str, old: str, new: str, label: str) -> str:
    count = code.count(old)
    if count != 1:
        raise RuntimeError(f"V4.2 patch expected one {label} block, found {count}")
    return code.replace(old, new, 1)


def apply_v42(code: str) -> str:
    old_sampler = '''class BalancedSourceClassBatchSampler(Sampler):
    """Every training batch contains all available source/class groups."""
    def __init__(self, df, batch_size, seed=2029):
        self.df = df.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.groups = {
            key: np.asarray(indices, dtype=int)
            for key, indices in self.df.groupby(["source", "label"]).groups.items()
        }
        self.keys = sorted(self.groups)
        if self.batch_size < len(self.keys):
            raise ValueError(f"batch_size={self.batch_size} is smaller than {len(self.keys)} source/class groups")
        self.steps = max(1, math.ceil(len(self.df) / self.batch_size))

    def set_epoch(self, epoch): self.epoch = int(epoch)
    def __len__(self): return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.steps):
            batch = []
            while len(batch) < self.batch_size:
                for key in self.keys:
                    if len(batch) >= self.batch_size: break
                    batch.append(int(rng.choice(self.groups[key])))
            rng.shuffle(batch)
            yield batch
'''

    new_sampler = '''class BalancedSourceClassBatchSampler(Sampler):
    """Balance source/class groups across micro-batches and accumulation windows.

    A micro-batch may be smaller than the number of source/class groups.  Each
    coverage cycle visits every available group at least once.  Within a
    micro-batch, group selection greedily favours different sources and
    different classes.  With the T4 profile (batch_size=2, grad_accum=2), the
    four groups are covered by two consecutive micro-batches before one
    optimizer update.
    """
    def __init__(self, df, batch_size, seed=2029):
        self.df = df.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.groups = {
            key: np.asarray(indices, dtype=int)
            for key, indices in self.df.groupby(["source", "label"]).groups.items()
        }
        self.keys = sorted(self.groups)
        if not self.keys:
            raise ValueError("BalancedSourceClassBatchSampler received an empty dataframe")
        self.grad_accum = max(1, int(CFG.get("grad_accum", 1)))
        self.coverage_batches = max(1, math.ceil(len(self.keys) / self.batch_size))
        self.steps = max(self.coverage_batches, math.ceil(len(self.df) / self.batch_size))

    def set_epoch(self, epoch): self.epoch = int(epoch)
    def __len__(self): return self.steps

    def _coverage_order(self, rng):
        remaining = list(self.keys)
        rng.shuffle(remaining)
        ordered = []
        while remaining:
            chunk = []
            while remaining and len(chunk) < self.batch_size:
                if not chunk:
                    pick_index = int(rng.integers(0, len(remaining)))
                else:
                    used_sources = {key[0] for key in chunk}
                    used_classes = {key[1] for key in chunk}
                    scores = []
                    for index, key in enumerate(remaining):
                        diversity = (
                            2.0 * float(key[0] not in used_sources)
                            + 2.0 * float(key[1] not in used_classes)
                        )
                        size_tiebreak = 1e-3 * math.log1p(len(self.groups[key]))
                        random_tiebreak = 1e-6 * float(rng.random())
                        scores.append((diversity + size_tiebreak + random_tiebreak, index))
                    pick_index = max(scores)[1]
                chunk.append(remaining.pop(pick_index))

            # A coverage cycle may end with a partial chunk.  Fill it using
            # other groups so DataLoader always receives a full micro-batch.
            while len(chunk) < self.batch_size:
                candidates = [key for key in self.keys if key not in chunk]
                if not candidates:
                    candidates = self.keys
                chunk.append(candidates[int(rng.integers(0, len(candidates)))])
            ordered.extend(chunk)
        return ordered

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        produced = 0
        while produced < self.steps:
            order = self._coverage_order(rng)
            for start in range(0, len(order), self.batch_size):
                keys = order[start:start + self.batch_size]
                batch = [int(rng.choice(self.groups[key])) for key in keys]
                rng.shuffle(batch)
                yield batch
                produced += 1
                if produced >= self.steps:
                    return
'''
    return _replace_once(code, old_sampler, new_sampler, "balanced sampler")
