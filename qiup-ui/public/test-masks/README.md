# Rabbit SLM test mask

`rabbit-slm-mask.png` is a browser-ready binary amplitude mask for testing the
web application's **Upload image** option.

- Source: `object_maps/assets/mask_rabbit.npy`
- Native image size: 254 × 297 px (width × height)
- White (`255`): transmitting SLM region
- Black (`0`): blocked SLM region
- PNG SHA-256: `f738659e1b46362c271b5980972d3e5b3c5b35669863a94636aeac2ae24e6954`

The web application resizes this or any other uploaded raster to the selected
square simulation grid. It preserves the aspect ratio, centers the resized
content, and pads the remaining pixels with zero transmission.
