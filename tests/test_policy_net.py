"""RL policy net + BC + PPO plumbing tests (torch; skip without it)."""

import random

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from hsbg_coach.bg_env import BGEnv, N_ACTIONS, A_END
from ml.env_obs import encode_obs, N_TOKENS, POLICY_CTX_DIM
from ml.policy_net import PolicyNet, save_policy, load_policy, as_env_policy
from ml.tokens import token_dim

EMB = {"A": [1.0, 0.0], "B": [0.0, 1.0]}


def obs_from_env(seed=0):
    env = BGEnv(seed=seed)
    obs = env.reset(seed=seed)
    return env, obs


def test_encode_obs_shapes():
    env, obs = obs_from_env()
    toks, mask, zones, ctx = encode_obs(obs, EMB, byname={})
    assert toks.shape == (N_TOKENS, token_dim(EMB))
    assert mask.shape == (N_TOKENS,) and zones.shape == (N_TOKENS,)
    assert ctx.shape == (POLICY_CTX_DIM,)
    assert mask[7:7 + len(obs["shop"])].all()      # shop zone populated


def test_policy_forward_and_masking():
    net = PolicyNet(token_dim(EMB))
    env, obs = obs_from_env(seed=1)
    arrays = encode_obs(obs, EMB, byname={})
    legal = env.legal_mask(0)
    logits, value = net(*[torch.from_numpy(a).unsqueeze(0) for a in arrays])
    assert logits.shape == (1, N_ACTIONS) and value.shape == (1,)
    masked = PolicyNet.masked_logits(
        logits, torch.tensor([legal], dtype=torch.float32))
    probs = torch.softmax(masked, dim=-1)[0]
    for i, ok in enumerate(legal):
        if not ok:
            assert probs[i].item() == 0.0


def test_act_returns_legal_action_and_roundtrip(tmp_path):
    net = PolicyNet(token_dim(EMB))
    env, obs = obs_from_env(seed=2)
    arrays = encode_obs(obs, EMB, byname={})
    legal = env.legal_mask(0)
    a, logp, v = net.act(arrays, legal)
    assert legal[a] and logp <= 0.0

    path = str(tmp_path / "p.pt")
    save_policy(net, path)
    net2 = load_policy(path)
    a2, _, _ = net2.act(arrays, legal, greedy=True)
    a3, _, _ = net.act(arrays, legal, greedy=True)
    assert a2 == a3


def test_env_policy_wrapper_plays_full_episode():
    net = PolicyNet(token_dim(EMB))
    pol = as_env_policy(net, EMB, byname={})
    env = BGEnv(seed=3)
    obs = env.reset(seed=3)
    rng = random.Random(0)
    for _ in range(400):
        a = pol(obs, env.legal_mask(0), rng)
        obs, reward, done, info = env.step(a)
        if done:
            assert 1 <= info["placement"] <= 8
            return
    pytest.fail("episode did not terminate")


def test_bc_learns_to_imitate():
    from ml.bc import collect, train_bc
    demos = collect(3, EMB, byname={}, seed=0)
    assert len(demos) > 30
    net = train_bc(demos, EMB, epochs=8, seed=0, verbose=False)
    toks = torch.from_numpy(np.stack([d[0][0] for d in demos]))
    mask = torch.from_numpy(np.stack([d[0][1] for d in demos]))
    zones = torch.from_numpy(np.stack([d[0][2] for d in demos]))
    ctx = torch.from_numpy(np.stack([d[0][3] for d in demos]))
    legal = torch.from_numpy(np.stack([d[1] for d in demos]))
    acts = torch.tensor([d[2] for d in demos])
    with torch.no_grad():
        logits, _ = net(toks, mask, zones, ctx)
        logits = PolicyNet.masked_logits(logits, legal)
        acc = float((logits.argmax(-1) == acts).float().mean())
    assert acc > 0.5                                  # imitates the baseline


def test_rollout_and_gae_shapes():
    from ml.rl_common import rollout
    from ml.train_ppo import _gae
    net = PolicyNet(token_dim(EMB))

    def step(arrays, legal):
        return net.act(arrays, legal)

    traj = rollout(step, seed=7, emb=EMB, byname={}, shaping=1.0)
    n = len(traj["action"])
    assert n > 0 and len(traj["reward"]) == n
    assert 1 <= traj["placement"] <= 8
    adv = _gae(traj["reward"], traj["value"])
    assert adv.shape == (n,) and np.isfinite(adv).all()
