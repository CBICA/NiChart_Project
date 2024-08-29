
let module_1_cache = {};
let module_2_cache = {}; 
let module_3_cache = {};

let use_cached_module1_results = false;
let use_cached_module2_results = false;

export function getUseModule1Results() {
    return use_cached_module1_results;
}

export function setUseModule1Results(setting) {
    use_cached_module1_results = setting;
}

export function getUseModule2Results() {
    return use_cached_module2_results;
}

export function setUseModule2Results(setting) {
    use_cached_module2_results = setting;
}

export function getModule1Cache() {
    return module_1_cache
}

export function setModule1Cache(modified) {
    module_1_cache = modified;
}

export function getModule2Cache() {
    return module_2_cache;
}

export function setModule2Cache(modified) {
    module_2_cache = modified;
}

export function getModule3Cache() {
    return module_3_cache;
}

export function setModule3Cache(modified) {
    module_3_cache = modified;
}

