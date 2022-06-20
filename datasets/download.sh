echo "Downloading PeerRead..."
git clone https://github.com/allenai/PeerRead.git

echo "Downloading ScisummNet..."
wget https://cs.stanford.edu/~myasu/projects/scisumm_net/scisummnet_release1.1__20190413.zip
unzip scisummnet_release1.1__20190413.zip
rm scisummnet_release1.1__20190413.zip
rm -rf __MACOSX
