# Specify path to the directory containing the capture.py and myTeam files
PATHPACMAN='../'
RED='MDP.v4'
echo "DATA" > output.log
for t in baselineTeam
do
#  for var in alleyCapture bloxCapture crowdedCapture defaultCapture distantCapture fastCapture jumboCapture mediumCapture officeCapture strategicCapture testCapture tinyCapture
#  do
#      MAP='FIXED'
#      BLUE=$t
#      echo #MAP$var
#      echo $MAP$var >> output.log
#      echo $RED >> output.log
#      echo $BLUE >> output.log
#      # Specify which teams should compete:
#      # echo python2.7 $PATHPACMAN/capture.py -r $PATHPACMAN/$RED.py -b $PATHPACMAN/$BLUE.py -l $PATHPACMAN/layouts/$var -q --delay-step=0.001
#      python2.7 $PATHPACMAN/capture.py -r $PATHPACMAN/$RED.py -b $PATHPACMAN/$BLUE.py -l $PATHPACMAN/layouts/$var -q --delay-step=0.001 | tee -a output.log
#      echo "= = = = = = = = = = = = = = = =" >> output.log
#  done
  # Update MAP seed values in the curly braces below
  for var in 191 841 7296 9592 2305 4917 5331 6903 3438 6777 5405 535 2821 8692 5144 6311 2608 4331 6146 5702 4636 4417 8278 1160 7633 8951 4469 2746 5193 128 3914 5732 6395 7953 6937 5018 5756 9274 2482 1279 1498 2110 3022 6592 9440 5686 1006 4921 7673 2963 455 5778 2425 5951 3100 8029 141 3633 8874 7748 2159 598 7045 3510 1155 9359 9577 5997 3796 6316 7693 507 1708 362 1928 288 2062 8301 9058 2700 8022 9317 668 6989 4753 4280 2708 6119 1337 7379 46 5589 3946 5823 6263 9802 4574 37 1621 3622 1224 3280 9013 9809 7064 5630 4169 6246 5784 2436 8155 1160 58 100 9825
  do
      MAP='RANDOM'
      echo $MAP$var >> output.log
      echo $RED >> output.log
      echo $BLUE >> output.log
      BLUE=$t
      # Specify which teams should compete:
      # echo python2.7 $PATHPACMAN/capture.py -r $PATHPACMAN/$RED.py -b $PATHPACMAN/$BLUE.py -l $MAP$var
      python2.7 $PATHPACMAN/capture.py -r $PATHPACMAN/$RED.py -b $PATHPACMAN/$BLUE.py -l $MAP$var -q --delay-step=0.001 | tee -a output.log
      echo $MAP$var
      echo "= = = = = = = = = = = = = = = =" >> output.log
  done
done
#############################################################
# To run the file, specify parameters
# Navigate to the shell file directory and run by: ./PacManTest.sh
# Fixed layout:
# alleyCapture, bloxCapture, crowdedCapture, defaultCapture, distantCapture, fastCapture, jumboCapture, mediumCapture, officeCapture, strategicCapture, testCapture, tinyCapture
#############################################################
