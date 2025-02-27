// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef PUBLICKEY_H
#define PUBLICKEY_H

#include "context.cuh"

namespace heongpu
{
    /**
     * @brief Publickey represents a public key used for encrypting data in
     * homomorphic encryption schemes.
     *
     * The Publickey class is initialized with encryption parameters and
     * provides a method to access the underlying public key data. This key is
     * used in conjunction with the HEEncryptor class to encrypt plaintexts,
     * making them suitable for homomorphic operations.
     */
    class Publickey
    {
        friend class HEKeyGenerator;

      public:
        /**
         * @brief Constructs a new Publickey object with specified parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         */
        __host__ Publickey(Parameters& context);

        /**
         * @brief Returns a pointer to the underlying public key data.
         *
         * @return Data* Pointer to the public key data.
         */
        Data* data();

        /**
         * @brief Returns the size of the polynomial ring used in the
         * homomorphic encryption scheme.
         *
         * @return int Size of the polynomial ring.
         */
        inline int ring_size() const noexcept { return ring_size_; }

        /**
         * @brief Returns the number of coefficient modulus primes used in the
         * encryption parameters.
         *
         * @return int Number of coefficient modulus primes.
         */
        inline int coeff_modulus_count() const noexcept
        {
            return coeff_modulus_count_;
        }

        /**
         * @brief Default constructor for Publickey.
         *
         * Initializes an empty Publickey object. All members will have their
         * default values.
         */
        Publickey() = default;

        /**
         * @brief Copy constructor for creating a new Publickey object by
         * copying an existing one.
         *
         * This constructor performs a deep copy of the public key data,
         * ensuring that the new object has its own independent copy of the
         * data. GPU memory operations are handled using `cudaMemcpyAsync` for
         * asynchronous data transfer.
         *
         * @param copy The source Publickey object to copy from.
         */
        Publickey(const Publickey& copy)
            : ring_size_(copy.ring_size_),
              coeff_modulus_count_(copy.coeff_modulus_count_)
        {
            locations_.resize(copy.locations_.size(), cudaStreamLegacy);
            cudaMemcpyAsync(locations_.data(), copy.locations_.data(),
                            copy.locations_.size() * sizeof(Data),
                            cudaMemcpyDeviceToDevice,
                            cudaStreamLegacy); // TODO: use cudaStreamPerThread
        }

        /**
         * @brief Move constructor for transferring ownership of a Publickey
         * object.
         *
         * Transfers all resources, including GPU memory, from the source object
         * to the newly constructed object. The source object is left in a valid
         * but undefined state.
         *
         * @param assign The source Publickey object to move from.
         */
        Publickey(Publickey&& assign) noexcept
            : ring_size_(std::move(assign.ring_size_)),
              coeff_modulus_count_(std::move(assign.coeff_modulus_count_)),
              locations_(std::move(assign.locations_))
        {
            // locations_ = std::move(assign.locations_);
        }

        /**
         * @brief Copy assignment operator for assigning one Publickey object to
         * another.
         *
         * Performs a deep copy of the public key data, ensuring that the target
         * object has its own independent copy. GPU memory is copied using
         * `cudaMemcpyAsync`.
         *
         * @param copy The source Publickey object to copy from.
         * @return Reference to the assigned object.
         */
        Publickey& operator=(const Publickey& copy)
        {
            if (this != &copy)
            {
                ring_size_ = copy.ring_size_;
                coeff_modulus_count_ = copy.coeff_modulus_count_;

                locations_.resize(copy.locations_.size(), cudaStreamLegacy);
                cudaMemcpyAsync(
                    locations_.data(), copy.locations_.data(),
                    copy.locations_.size() * sizeof(Data),
                    cudaMemcpyDeviceToDevice,
                    cudaStreamLegacy); // TODO: use cudaStreamPerThread
            }
            return *this;
        }

        /**
         * @brief Move assignment operator for transferring ownership of a
         * Publickey object.
         *
         * Transfers all resources, including GPU memory, from the source object
         * to the target object. The source object is left in a valid but
         * undefined state.
         *
         * @param assign The source Publickey object to move from.
         * @return Reference to the assigned object.
         */
        Publickey& operator=(Publickey&& assign) noexcept
        {
            if (this != &assign)
            {
                ring_size_ = std::move(assign.ring_size_);
                coeff_modulus_count_ = std::move(assign.coeff_modulus_count_);

                locations_ = std::move(assign.locations_);
            }
            return *this;
        }

      private:
        int ring_size_;
        int coeff_modulus_count_;
        bool in_ntt_domain_;

        DeviceVector<Data> locations_;
    };

    /**
     * @brief MultipartyPublickey is a specialized class for managing public
     * keys in multiparty computation (MPC) settings.
     *
     * This class extends the `Publickey` class to include functionality
     * specific to MPC scenarios, such as managing a seed for deterministic key
     * generation across multiple participants. It integrates with the
     * `HEKeyGenerator` class to facilitate collaborative key generation.
     */
    class MultipartyPublickey : public Publickey
    {
        friend class HEKeyGenerator;

      public:
        __host__ MultipartyPublickey(Parameters& context, const int seed);

        /**
         * @brief Retrieves the seed value used for key generation.
         *
         * @return int The seed value.
         */
        inline int seed() const noexcept { return seed_; }

      private:
        int seed_;
    };

} // namespace heongpu
#endif // PUBLICKEY_H