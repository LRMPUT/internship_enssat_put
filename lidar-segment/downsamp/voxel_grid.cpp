#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <string>

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("Not enough args \n");
        printf("Command should be : ./voxel_grid <base file> <end file>");
        exit(0);
    }

    pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2());
    pcl::PCLPointCloud2::Ptr cloud_filtered(new pcl::PCLPointCloud2());

    // Fill in the cloud data
    pcl::PCDReader reader;
    pcl::PCDWriter writer;

    // for (size_t i = 1; i < argc; i++)
    //{
    std::string fileName = argv[1];
    reader.read(fileName, *cloud); // Remember to download the file first!

    std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height
              << " data points (" << pcl::getFieldsList(*cloud) << ")." << std::endl;

    // Create the filtering object
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud(cloud);
    printf("Leaf size : %f %f %f\n", sor.getLeafSize()[0], sor.getLeafSize()[1], sor.getLeafSize()[2]);
    sor.setLeafSize(0.005f, 0.005f, 0.005f); // Plus c'est proche de 0, moins les points seront supprimés
    sor.filter(*cloud_filtered);

    std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height
              << " data points (" << pcl::getFieldsList(*cloud_filtered) << ")." << std::endl;

    std::string newFileName = argv[2];
    writer.write(newFileName, *cloud_filtered,
                 Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(), false);
    //}

    return (0);
}