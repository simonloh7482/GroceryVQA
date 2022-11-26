#!/bin/bash

readfile="viz_frei_aeon_train.json"
writefile="viz_frei_aeon_train_bg.json"
echo [ > $writefile

reg="^([a-z]+)+([0-9]+)\.(jpg)$"

jq -c '.[]' $readfile | while read it;
do
    fname=$( jq -r '.image' <<< $it )
    if [[ $fname =~ $reg ]]
    then
        ctg="${BASH_REMATCH[1]}"
        num="${BASH_REMATCH[2]}"
        #ext="${BASH_REMATCH[3]}"
        ext="png"

        rtt="A"
        up_fname="$ctg$num$rtt.$ext"
        up_it=$( jq --arg up_fname "$up_fname" '.image |= $up_fname' <<< $it )
        jq '.' >> $writefile <<< $up_it
        sed -i '$ s/$/,/' $writefile
        
        rtt="B"
        up_fname="$ctg$num$rtt.$ext"
        up_it=$( jq --arg up_fname "$up_fname" '.image |= $up_fname' <<< $it )    
        jq '.' >> $writefile <<< $up_it
        sed -i '$ s/$/,/' $writefile

    else
        jq '.' >> $writefile <<< $it
        sed -i '$ s/$/,/' $writefile

    fi
done

sed -i '$ s/,//' $writefile
echo ] >> $writefile
