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
    var return_str = value;
    var found_paper = false;
    data.forEach(function(item) {
      console.log()
      if (!found_paper && item.paper != 'x') {
        if (item.paper_link)
          return_str = value + '&nbsp;<a class="method-text" href="' + item.paper_link + '" target="_blank">' + item.paper + '</a>&nbsp;';
        else
          return_str = value + '&nbsp;<span class="is-italic has-text-white is-size-7">' + item.paper + '</span>&nbsp;';
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
