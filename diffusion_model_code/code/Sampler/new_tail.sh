
function tail_new1 () {

    final_index=$((${#@} - 1))
    #fn=$@ #[final_index] 
    #echo $@[final_index]
    fn=`echo "${@: -1}"`
    out=`tail $@``[[ $(tail -c1 $fn) && -f $fn ]]&&echo ''>>$fn`    
    echo $out
}

function tail_new2 () {
    
    filename=`echo "${@: -1}"`
    tail $@
    echo `[[ $(tail -c1 $filename) && -f $filename ]]&&echo '\n'>>$filename`
}

function tail_new () {
    
    tail $@
    inputfile=`echo "${@: -1}"`
    if [ "$(tail -c1 "$inputfile"; echo x)" != $'\nx' ]; then
        echo "" 
    fi

}

tail_new -n 5 file

