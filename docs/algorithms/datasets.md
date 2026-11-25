# The datasets

The datasets of the lab's scans that the [guide](README.md) works on, one block each: what it
holds, how big it is, what was checked on it and what was not. A dataset is a zip on the
lab's share, in the folder `smc-datasets`, once the owner has copied it there (ask the owner
for the path: it is not written in this repository). Compare the zip's SHA-256 with the one
below, then unzip it under `local/` and run `smc dataset check`, as
[Get a dataset](README.md#get-a-dataset) says. The SHA-256 of a file is `shasum -a 256
<file>.zip` on macOS and Linux and `Get-FileHash -Algorithm SHA256 <file>.zip` in PowerShell.

Every dataset has two texts beside its frames: the `README.md` that smc generates (the layout
and the conventions) and a `NOTES.md` that a person wrote (what the generator cannot know: the
camera, where the positions come from, the detector, and what was seen on a sample of the
frames). A `checked/` folder holds the pictures of that sample. `manifest.json` does not list
`NOTES.md` and `checked/`, so a check ignores them.

## The session of 2026-11-24

All three come from one session at the Nikon Ti2-E (Prime 95B camera), on a Greiner 662892
24-well plate turned by 180 degrees on the stage. Wells B3, B4, C3 and C4 were scanned at the
10x, then the whole of C4 and the whole of C3 at the 40x. Every scan has `coverage` 1 and
`overlap` 0.1. The perfect focus system (PFS) was locked at the 10x and **off** at the 40x (a
fixed Z, focused by eye at the well's centre).

| Dataset | Frames | Detections | Folder | Zip | SHA-256 of the zip |
| --- | --- | --- | --- | --- | --- |
| `2026-11-24-10x-4wells` | 436: wells B4, B3, C3, C4 at the 10x, 1.1243 µm/px | 296: single 21, doublet 31, debris 244 | 1.74 GB | 1.37 GB | `555f861de21a6136fe60e787af0ac11409443c2f973dcb223db795a959a2bbc0` |
| `2026-11-24-40x-C4` | 1544: well C4 at the 40x, 0.2776 µm/px | 8704: single 5116, doublet 1455, debris 2133 | 6.16 GB | 3.84 GB | `c333f96b8560540e77962ec7d1220375bc282c09ae85369e6d63c20834cbf151` |
| `2026-11-24-40x-C3` | 1544: well C3 at the 40x, 0.2776 µm/px | 1768: single 433, doublet 115, debris 1220 | 6.16 GB | 3.82 GB | `be8725de16c01192e3c6f4a706f93f50a25eb3d5aba889508aebed1a4f6f69d1` |

Every frame carries its stage position (the readback at its centre), its place in its well and
its Z, and every dataset carries the scan's camera, so that the stage position of anything in a
frame can be computed again (`README.md`, *Units and conventions*). The detections are
one record per cell: its class, its score, its box in pixels in its frame and its position in
stage µm; a cell that two overlapping fields both saw is one record (`cells_table`).

### `2026-11-24-10x-4wells`

The 10x scan of four wells, 109 fields each, PFS locked on all of them. Its detections were
made **at the stand** on the day, with the cell model at its defaults. By eye (12 frames, by a
Claude session, not by a person): air bubbles at a well's wall are boxed as "doublet", most of
the small dots on the frames are not boxed, and the class counts are not cell counts.

### `2026-11-24-40x-C4`

The whole of well C4 at the 40x: 1544 fields of 402 x 404 µm. Its detections were made
**offline** on 2026-11-25, with the same model at its defaults (`smc run cells --offline`;
12 minutes on a laptop CPU). A 24-frame 40x scan of well B4 of the same session gave exactly the
same counts offline and at the stand. By eye (12 frames, by a Claude session, not by a person):
in the interior the boxes sit on round bodies that look like cells, mostly as "single"; the
class is not consistent (round bodies also boxed as "debris", overlapping boxes in the densest
frame), and the frames at the wall are half black.

### `2026-11-24-40x-C3`

The whole of well C3 at the 40x, made like C4 (12 minutes of detection). This well is mostly an
empty background: 778 of its 1544 frames have no detection, and 1220 of its 1768 detections are
"debris", many of them on small droplets or particles. By eye (12 frames, by a Claude session,
not by a person): the few round bodies with a dark rim are boxed "single" or "doublet", most of
the particles in the dense fields are not boxed, and the frames at the wall are half black.

### What none of them can tell you

- **The 10x to 40x offset of the stage is not measured.** The plate was registered with the 10x
  objective (residual 25.5 µm) and the 40x frames' positions were not moved to it: an absolute
  position carries that unknown difference. Do not compare a cell of the 10x scan with a cell
  of a 40x scan to better than that without measuring it.
- **The model was not measured on these frames.** There is no annotated reference, so no
  precision and no recall; what was looked at is a sample, by eye, and each `NOTES.md` says what
  it showed. At the 10x the model finds almost no cell where at the 40x it finds thousands
  (well C4: 2 single and 12 doublet at the 10x, 5116 single and 1455 doublet at the 40x), so
  the detections of the two objectives are not comparable.
- **The run document's `camera-calibration` stage is the 10x one**, in the 40x datasets as well.
  The camera of a scan is the one in the scan's own result (`Scan.matrix_um_per_px` and
  `Scan.pixel_size_um`), and it is the one the detections' positions were computed with.
- **The focus at the 40x is the one set by eye at the well's centre**, with PFS off: nothing
  says that every frame is in focus.
