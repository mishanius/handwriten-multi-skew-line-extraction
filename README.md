## setup
1. compile anigauss as shared .dll(windows) or .so (linux) 
2. compile GCO using the provided makefile - `make testMain`
3. store compiled `testMain.exe` in `gco` folder
4. store compiled `anigauss.dll` in `anigauss` folder

 
### Excecution :
To excecute the code you need to specify the following arguments: 
1.  `--image_path` - the path of the image to process
2.  `--extractor` - the type of the algorithm you want to use options are - `MultiSkew`,`ExtractLines`
3.  `--mask_path` - for the ExtractLines algorithm, if provide the algorithm will use a premade Lines mask and will skip 
     the phase of creating the mask 
4.  `--no_cache` - wont use or update the cache.
5.  `--reset_cache` - purge the cache, before execution. 
#### stright line extraction:
```text
python ExtractLines.py --image_path binary_hetero_doc.png --extractor ExtractLines
```

#### multi skew line extraction:
```text
python ExtractLines.py --image_path binary_hetero_doc.png --extractor MultiSkew
```  

## project overview 
### multi-skew line extraction, uses the following steps
1. `filter_document` -  uses anisotropic gaussian filter inorder to find the maximum response of each pixel and the angel that gives
this response. from this step we will get two metricies one for the max response and the other for th e max orientation.
this function is inside `LineExtractorBase`, the base class for all the extractors. 
2. `__niblack_pre_process` - applyies niblack filter on the image
3. `split_lines` - here we use number of diffarent steps:
   1. aproximate each blob line with peacewise linear function with 20 knots 
      >there is no efficient implimintation that supports setting the desired number of knots in python.
      to solve this issue we use UnivariateSpline a One-dimensional smoothing spline fit to a given set of data points.
      UnivariateSpline find a spline function with desired rank, it sets the number of knots according to a constant 
      factor named "smoothing factor" this factor is different for different problems. 
      in in-order to find the factor that will give us a spline with 20 +-3 knots the function 
      `find_spline_with_numberofknots` this function uses binary search in order to find the factor which gives us 
      the closest result. This function is very slow, this issue should be addressed in the future.
   2. according to the spline we found we latter split the blobs into intact lines and broken lines - no major 
   implimintation differances between the python and the matlabversion.
      >we use Plantcv library for dilation skeletonize and find branch points operations 
4. `compute_line_label_cost` - this function computes the cost of each label ofthe avaliable labels 
      (each line and broken line get a unique label) this function also uses `local_orientation_label_cost`, 
      I modified this function in the following way:
      >in this function for each broken line that we found in the split phase we do a dilation, in matlab 
      implimintation we remove all other lines and than we make a dilation on the whole image, in python this is very 
      costly operation, so instead we generate an ROI of the expected size of the blob after dilation and excecute the 
      dilation operation on this ROI only.
5. `post_process_by_mfr` - 
    1. `computeNsSystem` - generates the neighbooring system forthe gco
    2. `compute_lines_data_cost` - computes the cost of assining label x to site (pixel) y, no major implimintation changes
    2. `line_extraction_GC` - the python entery point to use the GCO solver.
        >the current pygco implimintation lacks the option to add labels cost, in order to use label cost I created 
        a c++ rutine which readed csv files with all the required data (including labels cost) and solves the energy 
        minimization problem. `line_extraction_GC` creates the desired csv files and invokes the c++ rutine, the result 
        is the assigned labels.
        `line_extraction_GC`  also works the usual way (invokes directly the gco solver), but currently using the set 
        label cost option will result in a segmentation fault. To control the implimintation type use the the 
        `should_solve_via_csv` flag.
6. `join_segments_skew` - After labeling this method will merge blob that together will form a blob line.
    >  in this implimintation we generate a graph where each vertex is a blob. each blob has an edge to all the other blobs,
       and the weight of each edge is the auclidian distance between the blobs.
       Leter we found a MST of the graph, unlike the implimintation in the matlab we cant set waights that are 0 and we 
       cant set the root vertex. to solve the first issue I added 1 to all the weights, this wont effect the MST. I didnt 
       adressed the second issue because the corrent implimintation in python wont allow it and because this doesnt 
       effect the final result in Kruskal algorithm.  
     
     
## Caching system
For each function that has the `cacheble` decorator there a cache support. 
1. Before the method runs we check if in the cache folder 
   there exists a `.npz` file with the methods name (and some index to indicate the number of the execution).
2. If there is a `.npz` file we load it, parse the results and return the results without running the actual method.
3. If there no such file, the method will be executed and the results will be stored in an `.npz` file for latter.

Each function which returns a one ore more numpy_arrays/primitives can be decorated with `@cacheble`

## Debug and logging
For logging I created a special decorator `timed(lgnm=None, agregated=False, log_max_runtime=False, verbose=False)`
1.  `lgnm` - the name of the function that will appear in the log
2.  `agregated` - if true will agregate and log the total running time of the method.
3.  `log_max_runtime` -  if true will  log the max running time of the method.
4.  `verbose` - if true will log each time the methods starts and finishes and will log the run time for each execution.

### Partial Image
To show the partial result of methods that return an image I created a decorator `partial_image(index_of_output, name, binarize_image=True)`
1. `index_of_output` - for methods that return only one output this should be 0. 
2. `name` - name of the method, this will be part of the headline in the partial result image.
3. `binarize_image` - will show a gray scale image 

### examples:
#### input image:
![alt text](https://github.com/mishanius/HandWritenDocsLineExtraction/blob/refactor-line-extractor/test/ms_25_short.png "input")


#### partial results:

![alt text](https://github.com/mishanius/HandWritenDocsLineExtraction/blob/refactor-line-extractor/partial_results/niblack.PNG "niblack")

![alt text](https://github.com/mishanius/HandWritenDocsLineExtraction/blob/refactor-line-extractor/partial_results/split_lines.PNG "split lines")

![alt text](https://github.com/mishanius/HandWritenDocsLineExtraction/blob/refactor-line-extractor/partial_results/post_process_by_mfr.PNG "post process by mfr")
 
![alt text](https://github.com/mishanius/HandWritenDocsLineExtraction/blob/refactor-line-extractor/partial_results/join_segments_skew.PNG "join_segments_skew")

![alt text](https://github.com/mishanius/HandWritenDocsLineExtraction/blob/refactor-line-extractor/partial_results/post_process_by_mfr_2.PNG "post process by mfr2")


![alt text](https://github.com/mishanius/HandWritenDocsLineExtraction/blob/refactor-line-extractor/partial_results/final_result.PNG "result")
    
     
