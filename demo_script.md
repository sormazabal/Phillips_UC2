# Voiceover script — `demo_no_voice.webm`

Video: 163 s / 2:43, 30 fps. Audience: clinical / Philips stakeholder.
Pacing target ~145 wpm (~400 words ≈ 165 s). Timestamps are cue points, not hard cuts.

---

**[0:00–0:10]**
This is the Coronary angiography vessel & stenosis localisation viewer. One angiography frame goes in, and the model does two jobs: outline the coronary vessels, and find the devices inside them.

**[0:10–0:20]**
The operator starts by choosing a trained model, then loads the frame — either dragged in from the desktop, or pulled straight from the study folder.

**[0:20–0:36]**
From here it's automatic. The model runs once over the frame and returns a vessel map plus a set of candidate detections, each carrying its own confidence score. 

**[0:36–0:58]**
These three sliders decide how much of the proposal you accept. Confidence sets how sure the model has to be before a device is shown. The overlap slider removes duplicate boxes drawn around the same object. And the mask slider controls how generously the vessel outline is filled in.

**[0:58–1:12]**
Left is the untouched frame, right is the same frame annotated. The green box marks the detected device, with its confidence beside it. The cyan line is the vessel boundary the model traced.

**[1:12–1:32]**
Because angiography frames vary in exposure, brightness and contrast can be adjusted independently of the model. This changes only what the eye sees — the detection itself is untouched, so tuning the image for readability can never quietly change the result.

**[1:32–1:52]**
No model is right every time, so the last word belongs to the user. Switching on manual mode hands the AI's output over as an editable starting point: every box it proposed becomes a shape you can grab, and the vessel mask becomes a layer you can paint.

**[1:52–2:16]**
There are four tools. Move and resize adjusts a box the model placed slightly off. New box adds a device it missed entirely. Brush add paints in vessel the model under-segmented, and brush erase takes back anything it over-called. Nothing is destructive — one click resets everything to the original AI output.

**[2:16–2:34]**
Here the box is nudged onto the device tip, and the mask is extended with a brush stroke where the vessel faded out. The correction stays a proposal until it is submitted, so in-progress strokes never leak into the saved result.

**[2:34–2:43]**
Submitted. The annotation updates, the record is flagged as operator-edited, and the whole thing exports as JSON — traceable, and ready for the next frame.

---

## Alignment reference

| Time | On screen |
|---|---|
| 0:00 | App loaded, checkpoint `(none / zero-shot)`, empty uploader |
| 0:08–0:16 | CathAction checkpoint selected; `1.jpg` uploaded |
| 0:16–0:56 | Inference running; IoU 0.50→0.54, mask →0.54, confidence 0.50→0.45 |
| 0:56–1:12 | Original / Annotated columns render — green `device 1.00` box, cyan contour |
| 1:12–1:28 | Brightness dragged to 25, contrast to 1.30 |
| 1:32 | Manual mode toggled on |
| 1:36–2:14 | Canvas loading (~40 s of dead air), Submit disabled |
| 2:16–2:24 | Canvas ready; box dragged with Move / resize |
| 2:32 | Brush add — stroke painted around the device |
| 2:38–2:43 | "Corrections submitted", overlay relabels to `device manual` |

## Notes

- 1:36–2:16 is canvas load time. The `[1:52–2:16]` block exists to cover it. If that wait is trimmed in editing, cut that block and stretch `[1:32–1:52]`.
- Check the rendered TTS once against the video; if it runs past 2:43, trim `[0:36–0:58]` and `[1:52–2:16]` first — they are the two longest and least time-critical.


#Full:
This is the Coronary angiography vessel & stenosis localisation viewer. One angiography frame goes in, and the model does two jobs: outline the coronary vessels, and find the devices inside them.
The operator starts by choosing a trained model, then loads the frame.
From here it's automatic. The model runs once over the frame and returns a vessel map plus a set of candidate detections, each carrying its own confidence score. 
These three sliders decide how much of the proposal you accept. Confidence sets how sure the model has to be before a device is shown. The overlap slider removes duplicate boxes drawn around the same object. 
Left is the untouched frame, right is the same frame annotated. The green box marks the detected device, with its confidence beside it. The cyan line is the vessel boundary the model traced.
Because angiography frames vary in exposure, brightness and contrast can be adjusted independently of the model. This changes only what the eye sees — the detection itself is untouched, so tuning the image for readability can never quietly change the result.
No model is right every time, so the last word belongs to the user. Switching on manual mode hands the AI's output over as an editable starting point: every box it proposed becomes a shape you can grab, and the vessel mask becomes a layer you can paint.
There are four tools. Move and resize adjusts a box the model placed slightly off. New box adds a device it missed entirely. Brush add paints in vessel the model under-segmented, and brush erase takes back anything it over-called. Nothing is destructive — one click resets everything to the original AI output.
Here the box is nudged onto the device tip, and the mask is extended with a brush stroke where the vessel faded out. The correction stays a proposal until it is submitted, so in-progress strokes never leak into the saved result.
Submitted. The annotation updates, the record is flagged as operator-edited, and the whole thing exports as JSON.