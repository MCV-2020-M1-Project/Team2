#M1 - W2 (Group 2)

```
$ python m1_w2.py -h:
usage: m1_w2.py [-h] -t TASK -src SRC [-d DESCRIPTOR] [-lvl LEVEL] [-c CSP]
                [-ch1 CH1] [-ch2 CH2] [-bm BCKG_METHOD] [-m MEASURE]
                [-bbdd BBDD] [-plot PLOT] [-store STORE] [-k K]

optional arguments:
  -h, --help            show this help message and exit
  -t TASK, --task TASK  number of the task to execute: 1-6
  -src SRC, --source SRC
                        path to the folder with the images to analyse (qsd1,
                        qsd2, qtd1 or qtd2)
  -d DESCRIPTOR, --descriptor DESCRIPTOR
                        descriptor name, possible descriptors: ('1D_hist',
                        '2D_hist', '3D_hist')
  -lvl LEVEL, --level LEVEL
                        level of the multiresolution histograms, must be an
                        integer
  -c CSP, --csp CSP     color space, possible color spaces: ('RGB', 'HSV',
                        'CieLAB', 'YCbCr')
  -ch1 CH1, --channel1 CH1
                        channel selected to compute the 2D hist -> 0, 1 or 2
                        (Respetively b,g,r in RGB; l,a,b in CieLAB; Y,Cr,Cb in
                        YCrCb; H,S,V in HSV)
  -ch2 CH2, --channel2 CH2
                        channel selected to compute the 2D hist -> 0, 1 or 2
                        (Respetively b,g,r in RGB; l,a,b in CieLAB; Y,Cr,Cb in
                        YCrCb; H,S,V in HSV)
  -bm BCKG_METHOD, --bkcg_method BCKG_METHOD
                        method name, possible methods: ('msc', 'mcst', 'mcck',
                        'canny', 'watershed')
  -m MEASURE, --measure MEASURE
                        measure name, possible measure: ('eucl', 'l1', 'x2',
                        'h_inter', 'hell_ker', 'corr', 'chisq')
  -bbdd BBDD, --bbdd BBDD
                        path to the folder which contains the bbdd images
  -plot PLOT, --plot PLOT
                        allows plotting the results from the tasks
  -store STORE, --store STORE
                        stores the results from the tasks in the results
                        folder (see documentation)
  -k K, --k K           K value for MAP@K

```

Commands to execute different tasks:
- Task 2 -> 
```
python m1_w2.py -t 2 -bm msc -m l1 -src <PATH_TO_QSD2_w1> -bbdd <PATH_TO_BBDD> -d "3D_hist" -c "RGB" -store True -lvl <LVL>
```
- Task 3
```
python m1_w2.py -t 3  -m l1 -src <PATH_TO_QSD1_w2> -bbdd <PATH_TO_BBDD>  -lvl <LVL> -k <K>
```
- Task 4 -> get IoU
```
python m1_w2.py -t 4  -src <PATH_TO_QSD1_w2>S
```

- Task 6 
```
python m1_w2.py -t 6  -m l1 -src <PATH_TO_QSD2_w2> -bbdd <PATH_TO_BBDD> -lvl <LVL> -k <K>
```
