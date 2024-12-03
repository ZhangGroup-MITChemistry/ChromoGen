This code in this directory is _almost_ a direct copy of [this](https://github.com/BogdanBintu/ChromatinImaging/blob/master/CommonTools/PostAnalysisTools.py) file within the [ChromatinImaging GitHub repo](https://github.com/BogdanBintu/ChromatinImaging) associated with Bogdan Bintu's [2018 paper](https://doi.org/10.1126/science.aau1783), whose Fig. 4 results inspired our own Figs. 4a (bottom) and 4b. 

The _only_ changed is that we used Python's `2to3` command line utility to convert the code from Python 2 to Python 3 code, i.e.
```bash
2to3 -w PostAnalysisTools.py
```