from bench.workloads.generate import generate_trace, load_workload


def test_trace_replay_is_seeded_exact_length_and_scheduled():
    config = load_workload(__import__("pathlib").Path("configs/workloads/chat.yaml"))
    config["seed"] = 44
    corpus = list(range(100, 130))
    a = generate_trace(config, corpus, duration_s=20, rate_rps=3, max_requests=20,
                       prompt_tokens=420, output_tokens=8)
    b = generate_trace(config, corpus, duration_s=20, rate_rps=3, max_requests=20,
                       prompt_tokens=420, output_tokens=8)
    assert a == b
    assert len(a) == 20
    assert all(len(row["prompt_ids"]) == 420 and row["max_tokens"] == 8 for row in a)
    assert all(x["t_offset_ms"] < y["t_offset_ms"] for x, y in zip(a, a[1:]))
    assert all(row["req_id"] == f"W-chat-44-{i:06d}" for i, row in enumerate(a))
    groups = [(row["prefix_group"], row["prompt_ids"][:400]) for row in a
              if row["prefix_group"] is not None]
    assert groups
    assert all(prefix == groups[0][1] for group, prefix in groups
               if group == groups[0][0])
