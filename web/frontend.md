# npm and nodeJS
`lite-server` is a node module that only needed in dev phase, just like `browser-sync`.
```bash
npm install lite-server --save-dev
npm run lite-server
```

# Bootstrap4

### install with npm
downloading bootstrap
```bash
npm install bootstrap@4.0.0 --save
npm install jquery@3.3.1 popper.js@1.12.9 --save
```

### startup code
```html
<!-- Required meta tags within <head></head> -->
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
	<meta http-equiv="x-ua-compatible" content="ie=edge">

	<link rel="stylesheet" href="node_modules/bootstrap/dist/css/bootstrap.min.css">

<!-- js files to put at the bottom of the body tag -->
	<script src="node_modules/jquery/dist/jquery.slim.min.js"></script>
    <script src="node_modules/popper.js/dist/umd/popper.min.js"></script>
    <script src="node_modules/bootstrap/dist/js/bootstrap.min.js"></script>
```

### Grid system
There are five types of grid prefix: `.col-`, `.col-sm-`, `.col-md-`, `.col-lg-`, `.col-xl-`  
- `.col-` will always behave horizontal when the screen is extra small.
- `.col-<size>-` applies to all the screen sizes that equals or larger than the left threshold of `<size>` range.
- Margins
	- only `.col-` has no margin on two sides. 
	- `.col-<size>-` will have a fixed size (this value == left threshold) in the whole `size` range. 
- small gutter (white space) will be left between cols
- nestable

**Automatic col**  
Use `.col-sm` divs without indicating the numbers, and the grid system will automatically figure out the number of width for each div. 

Additionally, `order-sm-last`, `order-sm-first`, `order-sm-1`, ..., `order-sm-12` can be used to reorder the contents when the screen size changes. e.g. `<div class="col-sm-5 order-sm-last"></div>`

**Vertical Alignment**:  
For rows, `<div class="row align-items-center"></div>` will align the col center in y axis within the particular row.  
For cols, `<div class="col-12 col-sm-4 align-self-center"></div>` will align the content center corresponds only to the col. 

**Horizontal Alignment**:  
`<div class="row justify-content-center"></div>` divs will be centered in the particular row.  
In a special case, the `.col-auto` div will automatically determine the width number based on the content, and this is a good place to use `.justify-content-center` 

**Column Offsets**:  
Right shift the div: `<div class="col-sm-4 offset-sm-1"></div>`

**Nesting Columns**:  
Use `.row` within `.col`

### Headers and footers
**jumbotron**
`<header class="jumbotron"></header>` set the header apart from the rest of the page.

Both the header and footer are considered separated from the rest of the page. Thus, we should use separate container in the header and the footer, respectively. 

### Uncategorized
`<ul class="list-unstyled"></ul>` will remove the default bullitin points of ul.

`.text-center` used to center the text