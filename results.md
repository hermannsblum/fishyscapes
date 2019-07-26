---
title: Results
# subtitle: This is the demo site for Bulma Clean Theme
layout: page
show_sidebar: false
hide_footer: true
---


<script type="text/javascript" src="https://unpkg.com/tabulator-tables@4.2.7/dist/js/tabulator.min.js"></script>


<div id="results-table" class="is-size-7"></div>

<script>
//define some sample data
 var tabledata = [
 	{ 'method': 'random',
    'score': 'random',
    'static_FPR@95%TPR': 0.949993572318587,
    'static_average_precision': 0.0248497776346683,
    'static_experiment': 1575,
    'webmar19_FPR@95%TPR': 0.9500225574931618,
    'webmar19_average_precision': 0.026488297604628192,
    'webmar19_experiment': 1606},
  { 'method': 'softmax',
    'score': 'max_prob',
    'static_FPR@95%TPR': 0.3982947745822254,
    'static_average_precision': 0.12880622671874592,
    'static_experiment': 1357,
    'webmar19_FPR@95%TPR': 0.3361808379422649,
    'webmar19_average_precision': 0.1767311356707011,
    'webmar19_experiment': 1607},
  { 'method': 'softmax',
    'score': 'entropy',
    'static_FPR@95%TPR': 0.39754239485650766,
    'static_average_precision': 0.15405296895432505,
    'static_experiment': 1357,
    'webmar19_FPR@95%TPR': 0.3340478727347005,
    'webmar19_average_precision': 0.23556427038305203,
    'webmar19_experiment': 1607},
  { 'method': 'kNN embedding',
    'score': 'density',
    'static_FPR@95%TPR': 0.21550699859458075,
    'static_average_precision': 0.4972033184990905,
    'static_experiment': 1470,
    'webmar19_FPR@95%TPR': 0.17436438430380427,
    'webmar19_average_precision': 0.48024416126156505,
    'webmar19_experiment': 1608},
  { 'method': 'kNN embedding',
    'score': 'class_density',
    'static_FPR@95%TPR': 1.0,
    'static_average_precision': 0.15772478732995796,
    'static_experiment': 1562,
    'webmar19_FPR@95%TPR': 1.0,
    'webmar19_average_precision': 0.20354741377180133,
    'webmar19_experiment': 1619},
 ];

 //create Tabulator on DOM element with id "example-table"
var table = new Tabulator("#results-table", {
 	//height:205, // set height of table (in CSS or here), this enables the Virtual DOM and improves render speed dramatically (can be any valid css height value)
 	data:tabledata, //assign data to table
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
         headerSortStartingDir:"desc"},
        {title:"FPR@95%TPR",
         field:"static_FPR@95%TPR",
         align:"right",
         sorter:"number",
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
         headerSortStartingDir:"desc"},
        {title:"FPR@95%TPR",
         field:"webmar19_FPR@95%TPR",
         align:"right",
         sorter:"number",
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
    return value;
  },
});
</script>
