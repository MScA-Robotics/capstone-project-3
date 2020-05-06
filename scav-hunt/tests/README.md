# Unit Tests for Scavenger Hunt

## Cone Model Tests

To test the cone model first download (or take pictures of) the cones you wish
to run the test_edgetpu script against.

[Gleacher Center cone images](https://uchicago.box.com/s/i4ri6wv0b74f0u6qdnafix60ttgpaja6)

Save within a cones directory in this folder with each color in it's own sub-directory

    tests/
    |- cones/
    |  |- blue/
    |  |  |- blue1.jpg
    |  |  |- blue2.jpg
    |  |- purple
    |  |  |- purple1.jpg

Then run the testing script

    python3 test_edgetpu_cone_detect.py

Output will be a dict of all the pictures tested with the associated x coordiate of that cone
(false indicates that color cone was not found in the picture)
and an overall accuracy score of all the pictures tested

    "cones/blue/blue1.jpg":False,
    "cones/blue/blue2.jpg":0.39788,
    "cones/purple/1.jpg":0.52405,
    Total Accuracy:  66.67%
