# Presentation material

Store 4D SU(3) notebooks, figures, and presentation drafts here. Generated
outputs should remain under the ignored `artifacts/` directory.

`downsample_report.ipynb` reads
`artifacts/4dsu3/downsample/L24_beta6p20_to_L12_beta5p80_cnn_v2/` and plots
the training and validation loss, then the plaquette, \(1\times 2\) rectangle,
\(2\times 2\) square, and \(|P|^2\) histograms for the native L12 reference,
CNN downsampling, and naive two-link blocking. The last section writes down
the stout smearing, SU(3) projection, and factor-two blocking used by the CNN.
For 300 matched configurations the fixed split is 180 train, 60 validation,
and 60 test. Run `scripts/downsample/train.py` first if that run directory is
not present.
