for scene_path in $(ls -d /import/sisyphus/cvgpool3/jschoenb/learned-semantic3d/scannet/scene*)
do
    scene_name=$(basename $scene_path)
    echo $scene_name

    if [ ! -d ${scene_path}/label ]; then
        unzip $scene_path/${scene_name}_2d-label.zip -d ${scene_path}
    fi

    if [ ! -d ${scene_path}/sensor ]; then
        /import/local/ssd1/jschoenb/dev/code/ScanNet/SensReader/sens \
            $scene_path/${scene_name}.sens $scene_path/sensor
    fi

    if [ -d ${scene_path}/converted ]; then
        continue;
    fi

    mkdir -p $scene_path/converted/
    mkdir -p $scene_path/converted/images
    mkdir -p $scene_path/converted/groundtruth_model

    python /import/local/ssd1/jschoenb/dev/code/learned-semantic3d/TSDF/convert_scannet.py \
        --scene_path $scene_path \
        --output_path $scene_path/converted \
        --label_map_path /import/sisyphus/cvgpool3/jschoenb/learned-semantic3d/scannet/scannet-labels.combined.tsv \
        --resolution 0.05

    python /import/local/ssd1/jschoenb/dev/code/learned-semantic3d/tv_flux_3d.py \
        --datacost_path $scene_path/converted/groundtruth_datacost.npz \
        --output_path $scene_path/converted/groundtruth_model \
        --label_map_path $scene_path/converted/labels.txt

    python /import/local/ssd1/jschoenb/dev/code/learned-semantic3d/TSDF/tsdf_fusion.py \
        --input_path $scene_path/converted/ \
        --output_path $scene_path/converted/datacost \
        --frame_rate 50 \
        --resolution 0.05

done

