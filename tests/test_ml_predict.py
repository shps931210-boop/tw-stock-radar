"""每日影子預測：模型存檔後讀回的預測要和訓練時一致；同一天只封存一次。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from app import ml_predict as mp
from app import ml_research as ml
from tests.test_ml_research import WalkForward


class Predict(unittest.TestCase):
    def test_export_and_predict_roundtrip(self):
        P, sigs, dates = WalkForward().panel(n_dates=12, n=80, signal=0.05)
        with tempfile.TemporaryDirectory() as t:
            meta = mp.export_models(P, use_lgbm=False, out=Path(t))
            self.assertEqual(set(meta["horizons"]), {"5", "20", "60"})
            M = mp.load_models(Path(t))
            X = P[P["date"] == sigs[-1]].copy()
            out = mp.predict(X, M)["20"]
            # 真實訊號在 score_growth：Ridge 的預測應該和它高度正相關
            self.assertGreater(np.corrcoef(out["ridge"].to_numpy(), X["score_growth"].to_numpy())[0, 1], 0.8)
            self.assertTrue(((out["logit"] > 0) & (out["logit"] < 1)).all())

    def test_feature_change_invalidates_old_model(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "ml.json").write_text(json.dumps({"feats": ["x"], "horizons": {}}), "utf-8")
            self.assertIsNone(mp.load_models(Path(t)))

    def test_seal_is_write_once(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            (out / "2026-10-16.json").write_text("{}", "utf-8")
            fake = mock.MagicMock()
            fake.__enter__.return_value.execute.return_value.fetchone.return_value = ["2026-10-16"]
            with mock.patch.object(mp, "load_models", return_value={"horizons": {}}), mock.patch.object(mp.db, "connect", return_value=fake), \
                 mock.patch.object(mp.db, "get_meta", return_value=None), mock.patch.object(mp.pb, "load") as load:
                self.assertIsNone(mp.seal(out=out))
                load.assert_not_called()  # 已封存：連資料都不讀
            self.assertEqual((out / "2026-10-16.json").read_text("utf-8"), "{}")


if __name__ == "__main__":
    unittest.main()
