PSet 3 involves coding in a script and import it in the ipynb. This is easiest done with Jupyter Notebook. 
To work on Problem Set 3 with Google Colab is tricker because we will need to upload utilization scripts to Google Drive and let Colab find it. But here is how if you insist:

1. Open Google Colab
	colab.research.google.com
2. In the pop-up window, select 'Upload' tab, click 'browse', choose 'pset3.ipynb'

3. In the side-bar on the left, choose the icon that looks like a folder, then click on the 'Mount Drive' icon (the dark one with a Google Drive icon embedded)

4. You should see a new code cell automatically added with two lines. Run it and a link will be in the output. Copy the link and open in another browser tab, and you will see an authorization code. Copy it and paste back into the blank in the cell output then press enter

5. A line should be printed that indicates the mounting is successful (if so). Open Google Drive and upload mlp.py and utils.py from the PSet3 directory somewhere easy to find

6. Insert a code cell, copy the scripts to wherever the Colab is mounted. If you have uploaded the files to the home directory, use:
	!cp drive/My\ Drive/utlis.py .
	!cp drive/My\ Drive/mlp.py .
The backslash cancels the space interpretation and don't forget the dot at the end which means 'to the current directory'. If you have copied the files elsewhere, add the subdirectories between 'My\ Drive' and filename.

7. Try to run the first code cell to see if the Colab can correctly import mlp.py and utils.py. 

8. You can work on the coding part now. Upload and copy the mlp.py everytime you change it.

A few things to note:
1. The images in question won't be able to show in this way. Refer to them in the PSet3 directory.
2. Perform the upload and copy step every time you manipulate the codes because there is no direct way to access and write the files in Colab mounting directory.
