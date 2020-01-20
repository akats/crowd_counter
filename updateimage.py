import os
import os.path
import time

username="anatoliy"
hostname="ad.24dec.org:55521"
passw="9zU0Mm1zZsQ="
srcFile="results.tar.gz"
dstFile="/results.tar.gz"

if os.path.exists('tmp.jpg'):
  os.remove('tmp.jpg')

while 1:
  if not os.path.exists(srcFile):
     time.sleep(1)
  else:
     os.rename(srcFile, "tmp.jpg")
     # os.system('pscp -pw ' + passw + ' tmp.jpg ' + username+'@'+hostname+':'+dstFile)
     os.system('wput tmp.jpg ftp://' + username+':'+passw+'@'+hostname+dstFile)
     os.remove('tmp.jpg')
