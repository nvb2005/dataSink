---
layout: post
title: Utilities
date : 2016-05-04
categories : utilities
---

## Reduce the size of mp4

I had an mp4 file which was of size `132 MB`. I wanted to reduce the size of the file so that I can share it over mail. I googled for the solution and came across various options like using ffmpeg or other tools. After trying with all the options what worked for me is conversion using `**avconv**`. Here is the exact command I used on a `Ubuntu 14.04`.

```bash
avconv -i input.mp4 -s 1280x960 -strict experimental output.mp4
```

I had to add the option `-strict experimental` as the tool showed warning when used without it. You can play with different resolution for `-s` option. With resolution of `1280x960` the size of mp4 reduced to `25 MB`. Lower resolution can give smaller size, however it will affect the video quality.

I haven't explored other options. If your in a hurry then avconv can get our work done without needing to know about complex details about video format.

## Updating gems, ruby, Jeklly

* [gem upgrade and install](http://stackoverflow.com/questions/13626143/how-to-upgrade-rubygems)
* [ruby update and switch](http://stackoverflow.com/questions/16222738/how-do-i-install-ruby-2-0-0-correctly-on-ubuntu-12-04)
* [bunlde install](http://stackoverflow.com/questions/4402819/download-all-gems-dependencies)

