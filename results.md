---
title: Results
# subtitle: This is the demo site for Bulma Clean Theme
layout: page
show_sidebar: false
hide_footer: true
---

<script type="text/javascript" src="https://unpkg.com/tabulator-tables@4.2.7/dist/js/tabulator.min.js"></script>

# Evaluation Method

All evaluations are measured on our evaluation servers with data that is entirely unknown to the methods, in order to resemble true anomaly and make it harder for methods to overfit our data generation processes. This means that submissions contain binaries that are run over different input data.

The 'Fishyscapes Web' dataset is updated every three months with a fresh query of objects from the web that are overlayed on cityscapes images using varying techniques for every run. Methods are especially tested on new datasets that are generated only after the method has been submitted to our benchmark.

# Metrics

We use Average Precision (AP) as the primary metric of our benchmark. It is invariant to data balance and we are therefore able to accurately compare methods regardless of how many pixels they label as anomaly.
The tested methods output a continuous score for every pixel. We compute the metrics over all possible thresholds that a binary classifier could compare the output value with. The Average Precision is therefore also independent to the threshold a binary classifier could use.

In order to highlight safety-critical applications, we also compute the False Positive Rate at 95% True Positive Rate (TPR@95%FPR). This resembles the False Positive Rate of a binary classifier that compares the output value of the method against a threshold and classifies all pixels as anomaly that are above the threshold. We take exactly that threshold that results in 95% True Positive Rate, because it is more important in safety-critical systems to catch all anomalies than reducing false positives.

# Benchmark Results

<div id="results-table" class="is-size-7"></div>

<script>

 //create Tabulator on DOM element with id "example-table"
var table = new Tabulator("#results-table", {
 	//height:205, // set height of table (in CSS or here), this enables the Virtual DOM and improves render speed dramatically (can be any valid css height value)
 	//layout:"fitColumns", //fit columns to width of table (optional)
  groupBy:"method",
 	columns:[ //Define Table Columns
	 	//{title:"Method", field:"method", width:150},
    {title:"Score", field:"score", width:150, headerSort:false},
	 	{//column group
        title:"FS Static",
        columns:[
        {title:"AP",
         field:"static_average_precision",
         align:"right",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"desc"},
        {title:"FPR@95%TPR",
         field:"static_FPR@95%TPR",
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
         field:"webmar19_average_precision",
         align:"right",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"desc"},
        {title:"FPR@95%TPR",
         field:"webmar19_FPR@95%TPR",
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
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"desc"},
        {title:"FPR@95%TPR",
         field:"webjun19_FPR@95%TPR",
         align:"right",
         sorter:"number",
         sorterParams:{alignEmptyValues: 'bottom'},
         headerSortStartingDir:"asc"},
        ],
    },
 	],
  initialSort:[{column:"static_average_precision", dir:"desc"}],
  groupToggleElement:false, //no toggeling
  groupHeader:function(value, count, data, group){
    //value - the value all members of this group share
    //count - the number of rows in this group
    //data - an array of all the row data objects in this group
    //group - the group component for the group
    var return_str = '<span class="method">' + value + '</span>';
    var found_paper = false;
    data.forEach(function(item) {
      console.log()
      if (!found_paper && item.paper != 'x') {
        if (item.paper_link)
          return_str = return_str + '<a class="method method-paper" href="' + item.paper_link + '" target="_blank">' + item.paper + '</a>';
        else
          return_str = return_str + '</span><span class="method method-paper">' + item.paper + '</span>';
        found_paper = true;
      }
    });
    console.log(return_str);
    return return_str;
  },
});

console.log('start fetching');
fetch('https://spreadsheets.google.com/feeds/cells/1fJy2tsru1Sza37IZGk3PqTGbpA_kTsE_QK5Ld2v65bc/1/public/full?alt=json').then(function(response) {
  console.log('stop fetching');
  response.json().then(function(data) {
    console.log('json ready');
    // read in lists of rows
    var rows = [];
    var rowData = [];
    for(var r=0; r<data.feed.entry.length; r++) {
      var cell = data.feed.entry[r]["gs$cell"];
      var val = cell["$t"];
      if (val.endsWith('%')) val = parseFloat(val.slice(0, -1));
      if (cell.col == 1) {
        if (cell.row != 1) rows.push(rowData);
        rowData = [];
      }
      rowData.push(val);
    }
    rows.push(rowData);
    console.log('found all rows');
    // map to lists of dictionaries
    var tabledata = [];
    for(var r=1; r<rows.length; r++) {
      var row = {};
      rows[0].forEach(function(key, i) {
        row[key] = rows[r][i];
      });
      tabledata.push(row);
    }
    console.log(tabledata);
    table.setData(tabledata);
  });
});
</script>
