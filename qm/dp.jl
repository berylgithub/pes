


"""
main caller as usual
"""
function main()
    # read gdb file:
    f = open(pwd()*"/qm/dsgdb9nsd_000001.xyz")
    lines = readlines(f)
    display(lines)
    
end

main()