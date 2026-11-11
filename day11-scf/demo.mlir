func.func @sum_even_below_four() -> i32 {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    %two = arith.constant 2 : i32
    %sum = scf.for %i = %c0 to %c4 step %c1
        iter_args(%acc = %zero) -> (i32) {
        %i32 = arith.index_cast %i : index to i32
        %rem = arith.remsi %i32, %two : i32
        %is_even = arith.cmpi eq, %rem, %zero : i32
        %next = scf.if %is_even -> (i32) {
        %added = arith.addi %acc, %i32 : i32
        scf.yield %added : i32
        } else {
        scf.yield %acc : i32
        }
        scf.yield %next : i32
    }

    return %sum : i32
}