# Manual GUI Acceptance Checklist — Adinn 4K Image Enhancer

RC checklist for the **packaged/installed** app (`dist\Adinn4KImageEnhancer\Adinn4KImageEnhancer.exe`,
or the installed Start-Menu shortcut). This is the manual, in-window pass the automated
HTTP-level tests cannot cover: real clicks, real drags, real mouse gestures, real dialogs.

## How to run

- Use the **packaged EXE** (not the dev server) so the native shell, native Save-As and
  desktop-only paths are what's exercised.
- One row per test. Result = `PASS` / `FAIL`. Observation = one short sentence — what you
  actually saw (optionally the measured number, e.g. job time).
- Use this baseline input set:
  - one small `.jpg` (e.g. `image_enhancer/originals/2.jpeg`)
  - one small `.png`
  - a 2-3 image batch
- Note the machine's device context at the top of each session (GPU present / CPU only).

## Environment

| Field | Value |
|---|---|
| Machine / OS | |
| GPU present | yes / no |
| App version | (from the Settings/about or installer name) |
| Tested by | |
| Date | |

## Checklist

| # | Test | Expected behavior | Result | Observation |
|---|---|---|---|---|
| 1 | Launch packaged EXE | App opens the splash screen, then the main window with the "Enhance Your Images to 4K" upload screen. No console window, no error dialog. | | |
| 2 | Upload one image | "Browse / Choose files" accepts one `.jpg`; the file appears as a thumbnail with a visible size; the Enhance button becomes enabled. | | |
| 3 | Drag/drop image | Dragging a `.jpg`/`.png` from Explorer onto the drop zone adds it as a thumbnail (same result as #2). | | |
| 4 | Multiple image upload | Selecting 2-3 files at once produces one thumbnail per file, all with sizes. | | |
| 5 | Thumbnail removal / add more | Each thumbnail has a remove (x) control; removing one drops only that file. After removal you can add more files and the set stays correct (no duplicates, no ghost thumbnails). | | |
| 6 | Enhance | Clicking Enhance starts a job; with a single image the app moves to the progress view. Job completes on GPU/CPU per the current device setting. | | |
| 7 | Progress state | During processing: job status/progress indicator updates, current file and stage labels advance, no frozen or blank state; completes within a reasonable window (tens of seconds per image on this machine). | | |
| 8 | Result display | On completion the result view appears: the enhanced image plus the original, a compare slider, adjustment sliders, board editor and export controls. | | |
| 9 | Before/after comparison | Dragging the compare slider splits the view between the original (weak/soft/noisy) and enhanced (sharp/de-noised/4K-longest-side) image; the slider reaches both extremes cleanly. | | |
| 10 | Brightness | Moving the Brightness slider up lightens the preview, down darkens it; preview updates without a network round-trip (no flicker). | | |
| 11 | Contrast | Moving Contrast changes the tonal range of the preview symmetrically. | | |
| 12 | Highlights | Increasing Highlights brightens bright areas more than shadows; decreasing darkens them, midtones/shadows stay close to unchanged. | | |
| 13 | Shadows | Increasing Shadows lifts the dark areas of the preview; decreasing pushes them down. | | |
| 14 | Saturation | Increasing Saturation intensifies colors; decreasing moves the preview toward grayscale (0 = fully desaturated). | | |
| 15 | Detail | Increasing Detail sharpens edges/detail in the preview; decreasing softens. No obvious ringing/overshoot artifacts at moderate values. | | |
| 16 | Reset adjustments | A Reset (defaults) control returns all sliders to neutral (0) and the preview back to the engine's plain output. | | |
| 17 | Independent adjustments per image (batch) | In a multi-image batch, set different values on image A and image B, then switch back and forth: each image keeps its own slider values; an export of A uses only A's values, and B only B's. | | |
| 18 | Board rectangle creation | In the board editor, click-and-drag on the photo creates a red rectangle over a sign/board area; the board count increments. | | |
| 19 | Board rectangle move | Grabbing inside an existing board rectangle drags it to a new spot; it stays put on release. | | |
| 20 | Board rectangle resize | Dragging a corner/edge handle resizes the rectangle; aspect/size tracks the pointer, no jump. | | |
| 21 | Delete board | A per-board delete control removes one board; the count decrements and the rectangle disappears. | | |
| 22 | Clear all boards | A "clear all" action removes every board rectangle for the current image at once. | | |
| 23 | Cancel board edit | Leaving the board editor with a "done/cancel/no" path (after drawing) abandons the in-progress/last edit and returns the board state to what it was before editing began. | | |
| 24 | Confirm board edit | Accepting/confirming the board edit keeps the rectangles; the board state persists and feeds the export. | | |
| 25 | PNG export | Save As with PNG selected writes a real `.png` file to the chosen folder; it opens cleanly and matches the adjusted preview. | | |
| 26 | JPG export | Save As with `.jpg` selected writes a JPEG viewable in the default viewer. | | |
| 27 | JPEG export | Save As with `.jpeg` selected writes a valid `.jpeg` (works, same as #26). | | |
| 28 | Include board outlines (export) | With "Include board outlines" on, the exported file has the red board outlines burned into the image. | | |
| 29 | Exclude board outlines (export) | With the flag off, the exported file has no outline overlay. | | |
| 30 | ZIP batch export | "Save ZIP" on a multi-image batch writes one `.zip` containing one image per input, each image keeping its own boards/adjustments only. | | |
| 31 | Recent results | After saving, the Recent page lists the saved file(s) (image or ZIP); entries survive a restart and open/point at the real saved files. | | |
| 32 | Open output directory | The Recent page/adjustment screen "open folder / show in Explorer" action opens Windows Explorer at the saved file's folder (selecting the file when it still exists). | | |
| 33 | Settings | The Settings page loads, shows the current processing device and its detected availability (e.g. "NVIDIA RTX ... detected"), and persists the selection across runs. | | |
| 34 | Auto device | With Auto selected, enhancement runs on the GPU when one is detected, CPU otherwise; "current device" shown matches what actually ran. | | |
| 35 | GPU device | Forcing GPU runs the job on CUDA (GPU model/detection line), completes with correct output. Disabled/handled gracefully when no GPU exists. | | |
| 36 | CPU device | Forcing CPU runs the job on the OpenVINO CPU path, completes with a valid (softer) output. | | |
| 37 | Error handling | Trigger an error (e.g. stop the app mid-job, or upload a non-image/too-large file): the UI shows a clear, non-crashing error message and recovers to the upload screen; the app process stays alive. | | |
| 38 | Close/reopen app | Closing the window exits cleanly (no crash, no lingering process); reopening loads the app with workspace/Recent still intact. | | |

## Failure handling

If any test shows `FAIL`: note the exact symptom, whether the app crashed/hung/misbehavior,
any console/event-log hint, and the image/job that reproduced it. Do not close out the
release while a `FAIL` above exists.