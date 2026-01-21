from pathlib import Path
import numpy as np
import pandas as pd

# Takes grids dict + outputs .npy and/or .csv
def save_grids(grids: dict[str, np.ndarray], season: str, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for level, mat in grids.items():
        stem = f"{level.replace(' ', '_')}_fg_{season}"
        np.save(out_dir / f"{stem}.npy", mat)
        pd.DataFrame(mat).to_csv(out_dir / f"{stem}.csv", index=False)