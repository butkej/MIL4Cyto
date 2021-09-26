# MIL4Cyto
Released code for the paper ['End-to-end Multiple Instance Learning for Whole-Slide Cytopathology of Urothelial Carcinoma'](https://proceedings.mlr.press/v156/butke21a/butke21a.pdf)
by **Joshua Butke**, Tatjana Frick, Florian Roghmann, Samir F. El-Mashtoly, Klaus Gerwert and Axel Mosig.

Accepted to and presented at MICCAI 2021 Workshop Computational Pathology [(COMPAY 2021)](https://www.examode.eu/compay2021/).

## Overview
This repo contains the Python code of our experiments, however there is no data included. Still, this implementation might serve as a starting point for those interested in applying Attention-based Multiple Instance Learning to problems of cytopathology.

## Citation
Please cite our paper, if this work is of use to you or you use the code in your research:
    @inproceedings{butke2021end,
        title={End-to-end Multiple Instance Learning for Whole-Slide Cytopathology of Urothelial Carcinoma},
        author={Butke, Joshua and Frick, Tatjana and Roghmann, Florian and El-Mashtoly, Samir F and Gerwert, Klaus and Mosig, Axel},
        booktitle={MICCAI Workshop on Computational Pathology},
        pages={57--68},
        year={2021},
        organization={PMLR}
    }

## Requirements
Packages: 
- Pytorch (>= 1.6.0)
- OpenCV (4.4.0)
- sklearn (0.23.0)
- matplotlib (3.3.0)

Hardware:
We used a cluster equipped with 4 NVIDIA V100 GPUs, which is reflected in `joshnet/custom_model.py` where blocks of layers are assigned to dedicated cards.


## Further Reading
I highly recommend to check out the original paper and implementation of **Ilse et al.** for Attention-based MIL, that can be found [here](https://github.com/AMLab-Amsterdam/AttentionDeepMIL), as well as the **Li et al.** paper ['Deep Instance-Level Hard Negative Mining Model for Histopathology Images'](https://arxiv.org/pdf/1906.09681.pdf). The second one introduced Hard Negative Mining that I adopted. However they never released any code, so I implemented their improvements as best as I could.

## Contact
If you have any questions you can contact me at joshua.butke@ruhr-uni-bochum.de, however we do not gurantee any support for this software.

### Acknowledgements
This work was supported by the Ministry for Culture and Science (MKW) of North RhineWestphalia (Germany) through grant 111.08.03.05-133974 and the Center for Protein Diagnostics (PRODI).
