---
layout: post
title: Python Modules
date : 2016-05-03
categories : python-modules
---

## xlsxwriter

Python module `xlsxwriter` can be used to create new Microsoft 
excel file. It is important to note that, xlsxwriter can only 
create new Microsoft excel file. It cannot be used to read 
existing excel file. Here we will see how this module can be 
used to create simple excel file.

```python

import xlsxwriter as xls

## Create an empty xlsx file called book.xlsx
book = xls.workbook("book.xlsx")
## add a sheet to the workbook, default name for the sheet is sheet1
sheet1 = book.add_worksheet()
## add another sheet, but this time lets rename it to "raw_data" 
sheet2 = book.add_worksheet("raw_data")
```

Once the workbook and worksheet are created, we can start adding data.

```python

## lets add a string to first cell
sheet1.write(0, 0, "Day")
## this can also be written by specifying the cell number
## for the 0, 0 case the cell is A1. xlsxwriter uses 0 based indexing
sheet1.write("A1", "Day")
```

It should ideally create a Microsoft excel file named "book.xlsx" file. 
Sometimes (for the reason not known) you won't see this file created. 
Close the workbook object explicitly to create "book.xlsx" file.

```python
book.close()
```

## HTMLParser

Python HTMLParser module can be used to remove HTML entities.   

```python
In [29]: import HTMLParser
In [30]: html_parser = HTMLParser.HTMLParser()

In [31]: html_parser.unescape("&lt; &amp; &quot; &apos; &gt;")
Out[31]: u'< & " \' >'

In [34]: html_parser.unescape("&quot;between quotes&quot; and 3 &gt; 1")
Out[34]: u'"between quotes" and 3 > 1'
```


## UnitTest module

Python `unittest` module

* Create a `Unittest class` and make it subclass of `unittest.TestCase`
* Write functions which tests expected behaviour of functions
* Add check or assertions
* Run the unittest  

```python
import unittest

def multiple_of_two(num):
    """
    returns true if num is multiple of two
    """
    return (num % 2) == 0

def str_len_less_than_five(str1):
    """
    returns true if str len is less than 5
    """
    return len(str1) < 5

class MyUnitTest(unittest.TestCase):
    """
    UnitTest class
    """
    def test_multiple_of_two(self):
        self.assertTrue(multiple_of_two(200))
        self.assertFalse(multiple_of_two(201))
        
    def test_str_len_less_than_five(self):
        self.assertTrue(str_len_less_than_five("abc"))
        self.assertFalse(str_len_less_than_five("abcdef"))

##
## Running test directly from your functions
##
testsuite = unittest.TestLoader().loadTestsFromTestCase(MyUnitTest)
unittest.TextTestRunner(verbosity=2).run(testsuite)
```

To run tests from bash, add the below line to the script which will be called only when we invoke the script.  

```python
##
## Testing from bash
##
if __name__ == '__main__':
    unittest.main()
```

If your script is saved as `myunittest.py`, then you can run the unittest by running the script

```bash
python myunittest.py -v
```

Once run, it will test the condition and will list the test which are passed and the tests which failed.

```bash
test_multiple_of_two (__main__.MyUnitTest) ... ok
test_str_len_less_than_five (__main__.MyUnitTest) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.005s

OK
```
