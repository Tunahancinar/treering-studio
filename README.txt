TreeRingAI v6_PROFILE - Intensity Profile-Based Version

This version does not use skeletonization.
It is better suited to thin, closely spaced annual rings.

Workflow:
Photograph
-> Ruler-based calibration
-> Measurement line
-> YCbCr conversion
-> CLAHE
-> Bilateral filtering
-> Wide measurement band
-> 1D intensity profile
-> Multiscale peak/minimum detection
-> Merge nearby points
-> Calculate measurements in mm

Installation:
cd "C:\Users\TunahanC\Desktop\AI\TreeRingAI_v6_PROFILE"
py -m pip install -r requirements.txt

Run:
py main.py --input "..\11_1-13" --output "..\Sonuclar_v6"

For higher sensitivity:
py main.py --input "..\11_1-13" --output "..\Sonuclar_v6_hassas" --band-width 51 --min-ring-mm 0.05 --sensitivity high

If too many points are detected:
py main.py --input "..\11_1-13" --output "..\Sonuclar_v6_dengeli" --band-width 41 --min-ring-mm 0.12 --sensitivity normal
