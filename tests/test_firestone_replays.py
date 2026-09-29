"""firestone_replays: listing record -> manifest entry, replay conversion, dedup."""

import base64
import gzip
import io
import json
import os
import zipfile
import zlib

import pytest

from hsbg_coach import firestone_replays as fr

XML = b'<?xml version="1.0" encoding="utf-8"?><HSReplay><Game buildNumber="253216"/></HSReplay>'


def _fc(board, turn=9):
    return base64.b64encode(zlib.compress(json.dumps({"turn": turn, "board": board}).encode())).decode()


def _record(rid, ts=1790000000000, orig=None, **kw):
    board = [{"cardID": "BG_B", "tags": {"ZONE_POSITION": 2, "ATK": 5, "HEALTH": 6}},
             {"cardID": "BG_A_G", "tags": {"ZONE_POSITION": 1, "PREMIUM": 1, "ATK": 10, "HEALTH": 12}}]
    rec = {"reviewId": rid, "originalReviewId": orig or f"orig-{rid}", "buildNumber": 253216,
           "playerRank": "6288", "playerCardId": "TB_BaconShop_HERO_10",
           "replayKey": f"hearthstone/replay/2026/9/29/{rid}.xml.zip",
           "creationDate": "2026-09-29T08:55:01.856Z", "creationTimestamp": ts,
           "bgsAvailableTribes": [17, 23, 126], "bgsBannedTribes": [20],
           "bgsAnomalies": [], "bgsCompArchetype": "pirate_discover",
           "finalComp": _fc(board)}
    rec.update(kw)
    return rec


def _zip(xml=XML):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("replay.xml", xml)
    return buf.getvalue()


def test_manifest_entry_matches_what_replay_states_reads():
    e = fr.to_manifest_entry(_record("r1"))
    assert e["reviewId"] == "r1" and e["buildNumber"] == 253216 and e["mmr"] == 6288
    assert e["placement"] == 1
    assert e["finalComp"]["turn"] == 9
    # ordered by ZONE_POSITION, golden from PREMIUM, keys replay_states compares on
    assert [(m["cardId"], m["golden"]) for m in e["finalComp"]["board"]] == [
        ("BG_A_G", True), ("BG_B", False)]
    assert [t["name"] for t in e["tribes"]["available"]] == ["MECHANICAL", "PIRATE", "ABERRATION"]
    assert e["tribes"]["banned"][0]["name"] == "BEAST"
    assert e["creationDate"] == "2026-09-29T08:55:01.856Z" and e["creationTimestamp"] == 1790000000000


def test_final_comp_accepts_plain_dict_and_empty():
    assert fr.decode_final_comp({"turn": 3, "board": []}) == {"turn": 3, "board": []}
    assert fr.decode_final_comp(None) == {}


def test_replay_bytes_zip_gzip_xml():
    assert fr.replay_bytes_to_xml(_zip()) == XML
    assert fr.replay_bytes_to_xml(gzip.compress(XML)) == XML
    assert fr.replay_bytes_to_xml(XML) == XML
    with pytest.raises(ValueError):
        fr.replay_bytes_to_xml(b"<html>403</html>")


def test_write_xml_gz_rejects_non_replay(tmp_path):
    with pytest.raises(ValueError):
        fr.write_xml_gz(b"<?xml version='1.0'?><Other/>", str(tmp_path / "x.xml.gz"))
    fr.write_xml_gz(XML, str(tmp_path / "ok.xml.gz"))
    assert gzip.open(tmp_path / "ok.xml.gz").read() == XML


def test_select_new_skips_known_ids_dedupes_and_orders_oldest_first():
    listing = [_record("new2", ts=3), _record("old", ts=1), _record("new1", ts=2),
               _record("new1", ts=2), _record("dup-of-known", ts=4, orig="known-orig"),
               _record("nokey", ts=5, replayKey=None)]
    known = {"old", "known-orig"}
    assert [g["reviewId"] for g in fr.select_new(listing, known)] == ["new1", "new2"]
    assert [g["reviewId"] for g in fr.select_new(listing, known, limit=1)] == ["new1"]


def test_known_ids_includes_original_ids_and_replay_folder(tmp_path):
    (tmp_path / "ondisk.xml.gz").write_bytes(b"")
    ids = fr.known_ids([{"reviewId": "a", "originalReviewId": "a0"}], [str(tmp_path)])
    assert ids == {"a", "a0", "ondisk"}


def test_append_to_manifest_keeps_shape_and_dedupes(tmp_path):
    path = str(tmp_path / "manifest.json")
    json.dump({"note": "keep me", "games": [{"reviewId": "a"}]}, open(path, "w"))
    assert fr.append_to_manifest(path, [{"reviewId": "a"}, {"reviewId": "b"}]) == 1
    data = json.load(open(path))
    assert data["note"] == "keep me" and [g["reviewId"] for g in data["games"]] == ["a", "b"]
    listed = str(tmp_path / "list.json")
    json.dump([{"reviewId": "x"}], open(listed, "w"))
    fr.append_to_manifest(listed, [{"reviewId": "y"}])
    assert json.load(open(listed)) == [{"reviewId": "x"}, {"reviewId": "y"}]


def test_fetch_new_downloads_only_unseen_and_survives_a_bad_replay(tmp_path):
    listing = [_record("have", ts=1), _record("good", ts=2), _record("bad", ts=3)]
    manifest = str(tmp_path / "manifest.json")
    json.dump({"games": [{"reviewId": "have"}]}, open(manifest, "w"))

    def get(url):
        if url == fr.PERFECT_GAMES_URL:
            return json.dumps(listing).encode()
        if "good" in url:
            return _zip()
        return b"<Error>AccessDenied</Error>"

    out = str(tmp_path / "raw")
    res = fr.fetch_new(out, manifest, get=get, pause_s=0, log=lambda *_: None)
    assert res["listed"] == 3 and res["selected"] == 2
    assert [e["reviewId"] for e in res["entries"]] == ["good"]
    assert [f["reviewId"] for f in res["failed"]] == ["bad"]
    assert sorted(os.listdir(out)) == ["good.xml.gz"]
    assert json.load(open(manifest))["games"] == [{"reviewId": "have"}]   # untouched
