---
title: Results
# subtitle: This is the demo site for Bulma Clean Theme
layout: page
show_sidebar: false
hide_footer: false
---

<script type="text/javascript" src="https://unpkg.com/tabulator-tables@4.2.7/dist/js/tabulator.min.js"></script>

# Evaluation Method

All evaluations are measured on our evaluation servers with data that is entirely unknown to the methods, in order to resemble true anomaly and make it harder for methods to overfit our data generation processes. This means that submissions contain binaries that are run over different input data.

The 'Fishyscapes Web' dataset is updated every three months with a fresh query of objects from the web that are overlayed on cityscapes images using varying techniques for every run. Methods are especially tested on new datasets that are generated only after the method has been submitted to our benchmark.

# Metrics

We use **Average Precision (AP)** as the primary metric of our benchmark. It is invariant to data balance and we are therefore able to accurately compare methods regardless of how many pixels they label as anomaly.
The tested methods output a continuous score for every pixel. We compute the metrics over all possible thresholds that a binary classifier could compare the output value with. The Average Precision is therefore also independent to the threshold a binary classifier could use.

In order to highlight safety-critical applications, we also compute the **False Positive Rate at 95% True Positive Rate** (<svg height="1em" viewBox="-2 0 120 51" version="1.1"><text style="font-size:40px;line-height:125%;letter-spacing:0px;fill:#363636;fill-opacity:1;" x="-4.5" y="38">FPR</text><text style="font-size:30px;line-height:125%;letter-spacing:0px;fill:#363636;fill-opacity:1;" x="78.6" y="50.5">95</text></svg>). This resembles the False Positive Rate of a binary classifier that compares the output value of the method against a threshold and classifies all pixels as anomaly that are above the threshold. We take exactly that threshold which results in 95% True Positive Rate, because it is important in safety-critical systems to catch all anomalies, and for this threshold then pick the method which has the lowest number of false positives.

For methods that cannot use pretrained segmentation models, but require a special loss, this training or retraining can decrease the performance of the semantic segmentation. We therefore also report the mean intersection over union **(mIoU) on the Cityscapes validation set**.

**Runtime** is measured in seconds as the total time it takes for a submission to load an image from disk, run inference, and write the results to disk. We measure this as an average over 5000 images on a NVIDIA GTX3090 TI. Slower methods will have higher runtime, but the exact measurements should not be mistaken with the pure inference time.

# Benchmark Results

<div id="results-table" class="is-size-7"></div>

<script>

var fpr95 = '<svg height="1.2em" style="margin-top: 2px" viewBox="-2 0 120 52" version="1.1"><text style="font-weight:bold;font-size:40px;line-height:125%;letter-spacing:0px;fill:#363636;fill-opacity:1;" x="-4.5" y="38">FPR</text><text style="font-weight:bold;font-size:30px;line-height:125%;letter-spacing:0px;fill:#363636;fill-opacity:1;" x="78.6" y="50.5">95</text></svg>';
var black_cross = '<svg enable-background="new 0 0 24 24" height="14" width="14" viewBox="0 0 24 24" xml:space="preserve"><path fill="#222222" d="M22.245,4.015c0.313,0.313,0.313,0.826,0,1.139l-6.276,6.27c-0.313,0.312-0.313,0.826,0,1.14l6.273,6.272  c0.313,0.313,0.313,0.826,0,1.14l-2.285,2.277c-0.314,0.312-0.828,0.312-1.142,0l-6.271-6.271c-0.313-0.313-0.828-0.313-1.141,0  l-6.276,6.267c-0.313,0.313-0.828,0.313-1.141,0l-2.282-2.28c-0.313-0.313-0.313-0.826,0-1.14l6.278-6.269  c0.313-0.312,0.313-0.826,0-1.14L1.709,5.147c-0.314-0.313-0.314-0.827,0-1.14l2.284-2.278C4.308,1.417,4.821,1.417,5.135,1.73  L11.405,8c0.314,0.314,0.828,0.314,1.141,0.001l6.276-6.267c0.312-0.312,0.826-0.312,1.141,0L22.245,4.015z"></path></svg>';
var black_tick = '<svg enable-background="new 0 0 24 24" height="14" width="14" viewBox="0 0 24 24" xml:space="preserve"><path fill="#222222" clip-rule="evenodd" d="M21.652,3.211c-0.293-0.295-0.77-0.295-1.061,0L9.41,14.34  c-0.293,0.297-0.771,0.297-1.062,0L3.449,9.351C3.304,9.203,3.114,9.13,2.923,9.129C2.73,9.128,2.534,9.201,2.387,9.351  l-2.165,1.946C0.078,11.445,0,11.63,0,11.823c0,0.194,0.078,0.397,0.223,0.544l4.94,5.184c0.292,0.296,0.771,0.776,1.062,1.07  l2.124,2.141c0.292,0.293,0.769,0.293,1.062,0l14.366-14.34c0.293-0.294,0.293-0.777,0-1.071L21.652,3.211z" fill-rule="evenodd"></path></svg>';

 //create Tabulator on DOM element with id "example-table"
var table = new Tabulator("#results-table", {
 	//height:205, // set height of table (in CSS or here), this enables the Virtual DOM and improves render speed dramatically (can be any valid css height value)
 	//layout:"fitColumns", //fit columns to width of table (optional)
  groupBy:"method",
 	columns:[ //Define Table Columns
	 	//{title:"Method", field:"method", width:150},
    {title:"Score", field:"score", width:150, headerSort:false},
    {title: 'Method Requirements',
     columns:[
       {title:'retraining',
        field:"requires_retraining",
        align:"center",
        formatter: 'lookup',
        formatterParams: {'FALSE': black_cross, 'TRUE': black_tick},
        headerSort:false},
       {title:'OoD Data',
        field:"requires_ood_data",
        align:"center",
        formatter: 'lookup',
        formatterParams: {'FALSE': black_cross, 'TRUE': black_tick},
        headerSort:false},
       {title:'runtime',
        field:"runtime",
        align:"right",
        headerSort:false}
    ]},
    {title: 'Cityscapes',
     columns:[
       {title:'mIoU&nbsp;&nbsp;&nbsp;&nbsp;',
        field:"cityscapes_miou",
        align:"right",
        sorter:"number",
        sorterParams:{alignEmptyValues: 'bottom'},
        headerSortStartingDir:"desc"}]},
    {//column group
        title:"FS Lost & Found",
        columns:[
        {title:"AP",
         field:"LaF_AP",
         align:"right",
         cssClass:"column-group-left",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"desc"},
        {title:fpr95,
         field:"LaF_FPR@95%TPR",
         align:"right",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"asc"},
        ],
    },
    {//column group
        title:"FS Static",
        columns:[
        {title:"AP",
         field:"static_AP",
         align:"right",
         cssClass:"column-group-left",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"desc"},
        {title:fpr95,
         field:"static_FPR@95%TPR",
         align:"right",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"asc"},
        ],
    },
    {//column group
        title:"FS Web Aug 2022",
        columns:[
        {title:"AP",
         field: "webaug22_AP",
         align:"right",
         cssClass:"column-group-left",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"desc"},
        {title:fpr95,
         field:"webaug22_FPR@95%TPR",
         align:"right",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"asc"},
        ],
    },
    {//column group
        title:"FS Web Jan 2022",
        columns:[
        {title:"AP",
         field: "webjan22_AP",
         align:"right",
         cssClass:"column-group-left",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"desc"},
        {title:fpr95,
         field:"webjan22_FPR@95%TPR",
         align:"right",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"asc"},
        ],
    },
    {//column group
        title:"FS Web April 2021",
        columns:[
        {title:"AP",
         field: "webapr21_AP",
         align:"right",
         cssClass:"column-group-left",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"desc"},
        {title:fpr95,
         field:"webapr21_FPR@95%TPR",
         align:"right",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"asc"},
        ],
    },
    {//column group
        title:"FS Web Oct. 2020",
        columns:[
        {title:"AP",
         field: "weboct20_AP",
         align:"right",
         cssClass:"column-group-left",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"desc"},
        {title:fpr95,
         field:"weboct20_FPR@95%TPR",
         align:"right",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"asc"},
        ],
    },
    {//column group
        title:"FS Web Jan. 2020",
        columns:[
        {title:"AP",
         field: "webjan20_AP",
         align:"right",
         cssClass:"column-group-left",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"desc"},
        {title:fpr95,
         field:"webjan20_FPR@95%TPR",
         align:"right",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"asc"},
        ],
    },
    {//column group
        title:"FS Web Sept. 2019",
        columns:[
        {title:"AP",
         field: "websept19_AP",
         align:"right",
         cssClass:"column-group-left",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"desc"},
        {title:fpr95,
         field:"websept19_FPR@95%TPR",
         align:"right",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"asc"},
        ],
    },
    {//column group
        title:"FS Web June 2019",
        columns:[
        {title:"AP",
         field: "webjun19_AP",
         align:"right",
         cssClass:"column-group-left",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"desc"},
        {title:fpr95,
         field:"webjun19_FPR@95%TPR",
         align:"right",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"asc"},
        ],
    },
    {//column group
        title:"FS Web March 2019",
        columns:[
        {title:"AP",
         field:"webmar19_AP",
         align:"right",
         cssClass:"column-group-left",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"desc"},
        {title:fpr95,
         field:"webmar19_FPR@95%TPR",
         align:"right",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"asc"},
        ],
    },
 	],
  initialSort:[{column:"LaF_AP", dir:"desc"}],
  groupToggleElement:false, //no toggeling
  groupHeader:function(value, count, data, group){
    //value - the value all members of this group share
    //count - the number of rows in this group
    //data - an array of all the row data objects in this group
    //group - the group component for the group
    var return_str = '<span class="method">' + value + '</span>';
    var found_paper = false;
    data.forEach(function(item) {
      if (!found_paper && item.paper != 'x') {
        if (item.paper_link)
          return_str = return_str + '<a class="method method-paper" href="' + item.paper_link + '" target="_blank">' + item.paper + '</a>';
        else
          return_str = return_str + '</span><span class="method method-paper">' + item.paper + '</span>';
        found_paper = true;
      }
    });
    return return_str;
  },
});

fetch('https://sheets.googleapis.com/v4/spreadsheets/1fJy2tsru1Sza37IZGk3PqTGbpA_kTsE_QK5Ld2v65bc/values/Sheet1?alt=json&key=AIzaSyCIN71ETlQNIF460oLLaZAmTI8OdaiSVqc').then(function(response) {
  response.json().then(function(data) {
    // first row gives keys
    columnKeys = data.values[0];
    // read in lists of rows
    var tabledata = [];
    var currentEntry = {};
    for(var r=1; r<data.values.length; r++) {
      for(var i=0; i<data.values[r].length; i++) {
        var val = data.values[r][i];
        if (val.endsWith('%')) val = parseFloat(val.slice(0, -1));
        currentEntry[columnKeys[i]] = val;
      }
      // filter out any row that has not defined a method name
      if (currentEntry['method']) tabledata.push(currentEntry);
      currentEntry = {};
    }
    table.setData(tabledata);
  });
});
</script>
<br><br>
Methods that are not attributed in the table are adaptations to semantic segmentation based on different related works. The method details are presented in the [benchmark paper](https://arxiv.org/abs/1904.03215).
