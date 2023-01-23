wget https://github.com/q3aql/aria2-static-builds/releases/download/v1.36.0/aria2-1.36.0-linux-gnu-64bit-build1.tar.bz2
tar -xf aria2-1.36.0-linux-gnu-64bit-build1.tar.bz2
rm aria2-1.36.0-linux-gnu-64bit-build1.tar.bz2
mv aria2-1.36.0-linux-gnu-64bit-build1/aria2c aria2c
rm -rf aria2-1.36.0-linux-gnu-64bit-build1/

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
rm awscliv2.zip

mv aws/dist dist
rm -rf aws
mv dist aws

echo --- add these to your bashrc then restart ---
#echo export PATH=\"${PWD}:${PWD}/aws:\$PATH\"

echo note: adding these to the path will not work because a certificate it missing. 
echo To run without certificate add the following.
echo alias aws=\"${PWD}/aws/aws --no-verify-ssl\"
echo alias aria2c=\"${PWD}/aria2c  --check-certificate=false\"
echo alias does not work inside other shell scripts though so 