sd1-dwtdct.png
===============

512 x 512 RGB positive control, generated using the unmodified public API of
invisible-watermark 0.2.0 by ShieldMnt (MIT license):
https://github.com/ShieldMnt/invisible-watermark

    encoder = WatermarkEncoder()
    encoder.set_watermark("bytes", b"StableDiffusionV1")
    encoded_bgr = encoder.encode(carrier_bgr, "dwtDct")
    cv2.imwrite("sd1-dwtdct.png", encoded_bgr)

Payload: 136 bits, hexadecimal 537461626c65446966667573696f6e5631
SHA-256: 964006325b0b0edc0bd33222a7fbbbfa1b0997654694412aeafbf1df6c8933bb

Both the original upstream decoder and the app's pinned decoder recover
129/136 matching bits. The existing application threshold is 126/136.
The deterministic carrier before embedding is a negative control. It is
recreated by test_open_watermark_detection.py, not stored as a second image.

The carrier is an original synthetic texture (NumPy RNG seed 1729); it contains
no user image, downloaded photograph, or generated model output. Only the
upstream DWT-DCT encoder was used to insert the known public signal. No neural
model was loaded or downloaded. The original fixture came from the v1.0.16
public-payload regression; it was reused unchanged for the orientation/alpha
regression so those tests cannot silently change the watermark embedding.

This fixture tests the public Stable Diffusion DWT-DCT payload only. It is not
a SynthID sample, and it provides no evidence of SynthID detection/removal.
