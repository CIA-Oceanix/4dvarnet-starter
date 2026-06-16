for ref in $(git for-each-ref --format='%(refname)' refs/heads/); do
  git cat-file -e "$ref" 2>/dev/null || { echo "BROKEN: $ref"; rm -f ".git/$ref"; }
done
