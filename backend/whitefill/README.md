# Whitefill: pure-white cubes and triangles → STL

Photos in, one printable STL per piece out, sized in millimetres. It runs fully on the laptop: no internet and no API key.

## What it uses and what it ignores

- **Used:** one piece colour per scan: white (default, strict untinted white) or yellow, red, orange, green, blue, pink (solid, saturated pieces on a plain background; the `colour` setting). Cubes and triangles are recognised and completed; in `complete` mode other shapes get a best-guess outline.
- **Ignored:** pieces of other colours, pieces speckled or patterned with another colour, other shapes unless in `complete` mode, pieces bigger than 120 mm or smaller than 5 mm, and pieces cut off by the photo edge. A background of the same colour as the pieces (e.g. choosing blue on a blue table) is treated as too big and skipped.

## What the STL fills

| Setting | The STL is |
|---|---|
| `fill="complete"` (page default) | the whole piece with its missing parts restored. Cubes and triangles are completed from their own edges; other white shapes (discs, blobs) are not skipped but get a **best-guess** outline (convex hull of what is visible), flagged `guessed` in the result |
| `fill="white"` (API default) | exactly the pure-white area of each piece, with nothing around it |
| `fill="missing"` | only the part broken off each piece: its full cube or triangle outline minus the white that is there |

Whenever a cube or triangle has a broken part, a **fixed photo** (`fixed_<time>.jpg`) is also saved: the whole photo with each missing part painted back in using that piece's own colour. It is a flat 2D repair, not a coloured 3D model, and only pure-white pieces are repaired.

On the page, **Focus region** lets you drag up to 12 boxes on a photo; only the pixels inside them are analysed.

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
