from polyarb.orderbook import OrderBook


def test_replace_sorts_levels():
    b = OrderBook(asset_id="x")
    b.replace(
        bids=[(0.40, 10), (0.42, 5), (0.41, 7)],
        asks=[(0.46, 8), (0.45, 3), (0.47, 12)],
        ts_ms=1,
    )
    assert b.best_bid().price == 0.42
    assert b.best_ask().price == 0.45
    assert [lv.price for lv in b.bids] == [0.42, 0.41, 0.40]
    assert [lv.price for lv in b.asks] == [0.45, 0.46, 0.47]


def test_apply_change_update_then_remove():
    b = OrderBook(asset_id="x")
    b.replace([(0.5, 10)], [(0.6, 10)], 1)
    b.apply_change("SELL", 0.6, 4, 2)
    assert b.best_ask().size == 4
    b.apply_change("SELL", 0.6, 0, 3)
    assert b.best_ask() is None


def test_apply_change_inserts_new_level():
    b = OrderBook(asset_id="x")
    b.replace([], [(0.6, 5)], 1)
    b.apply_change("SELL", 0.55, 3, 2)
    assert b.best_ask().price == 0.55
    assert b.best_ask().size == 3


def test_zero_size_levels_filtered_on_replace():
    b = OrderBook(asset_id="x")
    b.replace([(0.5, 0), (0.4, 5)], [(0.6, 5), (0.7, 0)], 1)
    assert [lv.price for lv in b.bids] == [0.4]
    assert [lv.price for lv in b.asks] == [0.6]
