
for k in * ; do 
    if [ -d $k ] ; then 
        if [[ $k == 'bintu_data' ]] ; then
            continue
	else
            git add $k/* -f
	fi
    else
        git add $k -f
    fi

done

git commit -m "Backing up reviewer stuff"
git push 

