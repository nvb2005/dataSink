---
layout: post
title:  "Data Cleaning"
date:   2016-06-03
categories: data-cleaning
---

## Removing non-ascii data from text

* Using `iconv` to convert from any format to any format

  ```bash
  iconv -c -f utf-8 -t ascii infile > outfile
  
  iconv -c -f <input format> -t <output-format> infile > outfile
  ```

  
## Some techniques to clean the data

* This [webpage](http://www.analyticsvidhya.com/blog/2014/11/text-data-cleaning-steps-python/) has some tips or suggestion for data cleaning using python
  
