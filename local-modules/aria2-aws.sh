wget https://github.com/q3aql/aria2-static-builds/releases/download/v1.36.0/aria2-1.36.0-linux-gnu-64bit-build1.tar.bz2
tar -xf aria2-1.36.0-linux-gnu-64bit-build1.tar.bz2
rm aria2-1.36.0-linux-gnu-64bit-build1.tar.bz2
mv aria2-1.36.0-linux-gnu-64bit-build1/aria2c aria2c
rm -rf aria2-1.36.0-linux-gnu-64bit-build1/

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip

mv aws/dist dist
rm -rf aws
mv dist aws
