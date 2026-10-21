# C2PA detection regression fixtures

These files are small local test inputs. Tests do not fetch network resources,
upload user images, load neural models, or claim to test a SynthID pixel detector.

## Upstream files

`C.jpg`, `CA.jpg`, `CACA.jpg`, and `libpng-test_with_url.png` were obtained on
2026-10-20 from the official Content Authenticity Initiative C2PA Rust SDK:

https://github.com/contentauth/c2pa-rs/tree/main/sdk/tests/fixtures

The SDK is dual licensed under the accompanying `LICENSE-MIT` and
`LICENSE-APACHE`. These generated JPEG fixtures contain signed C2PA stores;
`CA.jpg` uses two JPEG APP11 packets and `CACA.jpg` uses four, covering the
previous fragmented-store false negative. The test signing certificate is not
a production trusted issuer. A valid signature is tested separately from issuer
trust. `libpng-test_with_url.png` contains an external manifest reference in XMP;
its localhost URL must **never** be fetched by these tests.

SHA-256:

| File | SHA-256 |
| --- | --- |
| C.jpg | a2d14755db55de67a47c04090340d8266e892367be4104a45626d7a6fa6e9ffd |
| CA.jpg | e71bff58fc57640803e6e65f7534e2fb0c2f99018c85276cc14b30f04427cc76 |
| CACA.jpg | e735fe9e50057f8280baf96182a38d516a4acfb144da7f1e6aad32de44c76aea |
| libpng-test_with_url.png | 8ffdf61c6497696b4a18f8e04b416c749f7ad67f4a1529862f50231db86d1411 |

## Locally generated files

`generated-signed.png` and `generated-signed.webp` are 64 × 48 solid RGBA
images (alpha = 120) generated with Pillow and signed using `c2pa-python 0.38.0`
with the upstream SDK's **public test** `es256_certs.pem` / `es256_private.key`:

https://github.com/contentauth/c2pa-python/tree/main/tests/fixtures

They contain deliberately synthetic `trainedAlgorithmicMedia` and
`c2pa.watermarked.unbound` / SynthID **metadata declarations** for testing the
claim parser. The pixels contain **no real SynthID watermark**. Detection of a
declaration is not evidence of pixel watermark detection or of the truth of its
claim. The test's generator name explicitly identifies it as a regression
fixture. No private production credentials are present in this directory.
