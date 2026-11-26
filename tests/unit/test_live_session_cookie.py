"""The live smoke suite signs dashboard sessions itself: it must sign exactly like the web."""

from tests.e2e.test_live import session_cookie


def test_matches_the_dashboards_signature() -> None:
    # node: createHmac("sha256", "demo-secret").update("v1." + expires).digest("base64url"),
    # which is what createSession in web/lib/session.ts computes with Web Crypto
    expected = "1800003600000.KXmy672vVifQDxcYEtrW0Ns3mqAT0-afM3vGoUV0VxU"

    assert session_cookie("demo-secret", now_ms=1_800_000_000_000, hours=1) == expected


def test_another_secret_signs_differently() -> None:
    cookie = session_cookie("demo-secret", now_ms=0)
    assert session_cookie("other-secret", now_ms=0) != cookie
    assert cookie.split(".")[0] == str(3_600_000)
