alias squeue='squeue -o %6A%12P%17j%10u%10T%5C%11m%12M%N'
alias endl='printf "\n"'
alias sq='squeue -u $USER'
alias slinfo='reset ; squeue -p Odyssey ; endl ; sinfo -p Odyssey ; endl ; sq ; endl'
