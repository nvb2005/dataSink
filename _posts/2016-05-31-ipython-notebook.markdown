---
layout: post
title:  "IPython notebook"
date:   2016-05-31
categories: ipython-notebook
---

## Math Equation in IPython Notebook

To add latex equation in IPython Notebook  

```latex
%%latex
$$c = \sqrt{a^2 + b^2}$$
```

Once run this will render math equation.




## Ploting figure inline

To display matplotlib plots inline on IPython notebook

```ipython
%matplotlib inline
```



## IPython notebook to Reveal.js slides

I was wondering if it is possible convert `IPython Notebook` 
directly to Slides. I searched for it and came across 
this particular article. Many people don't know about it. 
But it is possible to convert `IPython Notebook` directly to 
Slides. Details are here in this 
[blog](http://www.damian.oquanta.info/posts/make-your-slides-with-ipython.html)


## Setting up IPython notebook server

Create hashed password

```ipython
In [1]: from IPython.lib import passwd

In [2]: passwd()
Enter password: 
Verify password: 
Out[2]: 'sha1:0f674d42f45d:3609cf71e6fe06eb927099ba6a121e2fac41de6a'
```

Create profiles and certificate file

```ipython
rb@ccpp:~$ ipython profile create myserver
[ProfileCreate] Generating default config file: u'/home/rb/.ipython/profile_myserver/ipython_config.py'
[ProfileCreate] Generating default config file: u'/home/rb/.ipython/profile_myserver/ipython_notebook_config.py'
[ProfileCreate] Generating default config file: u'/home/rb/.ipython/profile_myserver/ipython_nbconvert_config.py'
```

```bash
## this will create a ssl certificate which we need to add to the profile.
## Even with that your browser will complain.
rb@ccpp:~$ openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout notebookcert.pem -out notebookcert.pem
```
We create certificate file and ipython profile. This config has many fields which are commented.
Make the following changes to run the IPython server and make it listen to specific port.

```ipython
## uncommented and made the port to listen to 8888 which is default port
c.NotebookApp.port = 8888

## uncommented and addeded the password generated above
c.NotebookApp.password = u'sha1:0f674d42f45d:3609cf71e6fe06eb927099ba6a121e2fac41de6a'

## uncommented so that server doesn't open the browser
c.NotebookApp.open_browser = False

## The IP address the notebook server will listen on.
## It was earlier set to localhost. Notsure why it is set to * though                                                                                           
c.NotebookApp.ip = '*'

# The full path to an SSL/TLS certificate file.                                                                                                
c.NotebookApp.certfile = u'/home/rb/notebookcert.pem'
```

Once these changes were done, you can start the ipython server by specifying the profile.

```ipython
rb@ccpp:~$ ipython notebook --profile=myserver
```

This will start a server and will make it listen to the specified port.

Now, open your browser and use your domain name and point it to port.

```bash
https://<your domain ip here>:8888
```

Your browser will complain about certification. Add an exception and you will see a login page.
Enter the password and you will be taken to main notebook page.

![notebook_login]({{ site.baseurl }}/assets/ipython_notebook_server_remote_login.jpeg)

Select the existing ipython notebook or create new notebook.

![notebook_mainpage]({{ site.baseurl }}/assets/ipython_remote_server_main_page.jpeg)