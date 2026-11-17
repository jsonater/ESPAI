# Whitefill: pure-white cubes and triangles → STL

Photos in, one printable STL per piece out, sized in millimetres. It runs fully on the laptop: no internet and no API key.

## What it uses and what it ignores

- **Used:** pure white only, and only pieces shaped like a cube (a square or rectangle from above) or a triangle.
- **Ignored:** every other colour (cream, yellow, pink, grey, light blue), white mixed or speckled with another colour, other shapes (discs, blobs), pieces bigger than 120 mm or smaller than 5 mm, and pieces cut off by the photo edge.

## What the STL fills

| Setting | The STL is |
|---|---|
| `fill="white"` (default) | exactly the pure-white area of each piece, with nothing around it |
| `fill="missing"` | only the part broken off each piece: its full cube or triangle outline minus the white that is there |

Each part's height is its cube's width (for a triangle, the average side). Set `height_mm` to force a height.

## Getting true sizes

Print `markers/marker_50mm.png` at 100 % and check that the black square is 50 mm. Tape it flat beside the pieces. With the marker in view, tilted photos are flattened first and every photo is measured; a pixel is kept when at least half the photos agree. Without the marker, sizes come from the manual mm-per-pixel setting and are only a guess.

Take photos from straight above with even light. Light grey and white in shadow can look alike.

## Files

| File | Job |
|---|---|
| `colors.py` | decides which pixels are pure white |
| `pieces.py` | separates the white pieces and traces their edges |
| `shapes.py` | cube or triangle? rebuilds the full outline |
| `detect.py` | per photo: which pixels the part should fill |
| `marker.py` | marker scale and perspective correction |
| `mesh.py` | writes the watertight STL |
| `sheet.py` | draws the measurement sheet |
| `pipeline.py` | `analyze()`: runs everything and combines the photos |

The API is in `backend/whitefill_api.py`; the page is `whitefill.html`. Each scan is saved to `data/whitefill/batch_NNNN/scan_<time>/`, so no folder holds more than 20 items.

## Use

```sh
uvicorn backend.main:app --host 127.0.0.1 --port 8000    # full app, then open /whitefill
python whitefill_laptop.py                                # whitefill only, opens the page
python -m unittest backend.tests.test_whitefill -v        # tests
python tools/whitefill_accuracy.py                        # accuracy on test_photos/
```

Measured on the photo sets in `test_photos/` (computer-made scenes, straight and tilted): every white cube and triangle found, nothing else picked up, broken-part size error 0.37 mm on average and 1.6 mm at worst. Real photos will be less accurate.
