rule("musa")
    set_extensions(".mu")

    on_build_file(function (target, sourcefile, opt)
        import("utils.progress")

        local objectfile = target:objectfile(sourcefile)
        local includedirs = table.wrap(target:get("includedirs"))
        local defines = table.wrap(target:get("defines"))
        local args = {"-c", sourcefile, "-o", objectfile, "-std=c++17", "-fPIC"}

        for _, includedir in ipairs(includedirs) do
            table.insert(args, "-I" .. path.absolute(includedir))
        end
        for _, define in ipairs(defines) do
            table.insert(args, "-D" .. define)
        end

        os.mkdir(path.directory(objectfile))
        progress.show(opt.progress, "compiling.musa %s", sourcefile)
        os.vrunv("/usr/local/musa/bin/mcc", args)
        table.insert(target:objectfiles(), objectfile)
    end)
rule_end()

target("llaisys-device-musa")
    set_kind("object")
    add_rules("musa")
    add_includedirs("../include")
    add_linkdirs("/usr/local/musa/lib", {public = true})
    add_links("musart", {public = true})
    add_files("../src/device/musa/*.mu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-musa")
    set_kind("object")
    add_deps("llaisys-tensor")
    add_rules("musa")
    add_includedirs("../include")
    add_linkdirs("/usr/local/musa/lib", {public = true})
    add_links("musart", "mublas", {public = true})
    add_files("../src/ops/*/musa/*.mu")

    on_install(function (target) end)
target_end()
