from bench.analyze.metrics import analyze, peak_open_streams, request_metrics


def record(req_id, send_ms, token_ms, *, status=200, prompt=10, done_prompt=10,
           expected=3, offset=0, error=None):
    events = [{"at_ns": round(t * 1e6), "index": i + 1}
              for i, t in enumerate(token_ms)]
    done = ({"at_ns": round((token_ms[-1] + 1) * 1e6),
             "finish_reason": "length", "usage": {"prompt_tokens": done_prompt,
                                                    "completion_tokens": len(events)}}
            if token_ms and status == 200 else None)
    return {"req_id": req_id, "send_ns": round(send_ms * 1e6),
            "t_offset_ms": offset, "generator_lag_ms": 0.5,
            "expected_prompt_tokens": prompt, "expected_output_tokens": expected,
            "ignore_eos": True, "token_events": events, "done": done,
            "status_code": status, "error": error}


def test_ttft_tpot_itl_are_client_observed_and_per_request():
    row = request_metrics(record("a", 0, [600, 650, 720]))
    assert row["ttft_ms"] == 600
    assert row["e2e_ms"] == 720
    assert row["itl_ms"] == [50, 70]
    assert row["tpot_ms"] == 60
    assert row["good"] is True
    late = request_metrics(record("b", 0, [1001, 1010, 1020]))
    assert late["good"] is False


def test_goodput_counts_individual_requests_and_rejections_separately():
    good = record("good", 0, [500, 550, 600], offset=1000)
    slow = record("slow", 0, [100, 200, 300], offset=2000)
    reject = record("reject", 0, [], status=429, offset=3000, error="overload")
    warmup = record("warmup", 0, [100, 120, 140], offset=100)
    summary = analyze([good, slow, reject, warmup], duration_s=5, warmup_s=1)
    assert summary["offered_requests"] == 3
    assert summary["completed_requests"] == 2
    assert summary["good_requests"] == 1
    assert summary["goodput_rps"] == 0.25
    assert summary["rejected_requests"] == 1
    assert summary["rejection_rate"] == 1 / 3
    assert summary["errors"] == 0
    assert summary["itl_p50_ms"] == 75


def test_count_mismatch_invalidates_result_and_one_token_has_no_tpot():
    mismatch = record("wrong", 0, [100, 120, 140], done_prompt=9)
    assert request_metrics(mismatch)["completed"] is False
    one = record("one", 0, [500], expected=1)
    result = request_metrics(one)
    assert result["tpot_ms"] is None
    assert result["good"] is True


def test_peak_open_streams_uses_received_headers_not_scheduled_sends():
    a = record("a", 0, [100, 120, 140])
    b = record("b", 0, [110, 130, 150])
    a["response_headers_ns"] = 10_000_000
    b["response_headers_ns"] = 20_000_000
    assert peak_open_streams([a, b]) == 2
